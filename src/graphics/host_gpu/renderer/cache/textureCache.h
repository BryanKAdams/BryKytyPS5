#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/dccClearResolver.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
struct TextureCacheTestAccess;

class TextureCache {
public:
	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };

	struct ImageDesc {
		ImageInfo     info;
		ImageViewInfo view_info;
		BindingType   type = BindingType::Texture;
	};

	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	// When the lookup finds exactly one image with the same backing, *unique_generation receives
	// the image set's generation, else 0: until an image is registered or unregistered, the same
	// request finds the same image, and RefindImage does the rest of FindImage for it.
	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false,
	                                      uint64_t* unique_generation = nullptr);
	// FindImage's bookkeeping for a request FindImage resolved to id under generation (see
	// unique_generation). Returns false, doing nothing, when the image set changed since.
	[[nodiscard]] bool          RefindImage(ImageId id, uint64_t generation, const ImageDesc& desc,
	                                        uint32_t metadata_base_layer);
	[[nodiscard]] uint64_t      ImageSetGeneration() const noexcept {
		return m_image_set_generation.load(std::memory_order_acquire);
	}
	// Whether FindTexture for a sampled texture that returned a view while the image set had this
	// generation would now only touch the image and return that view again: the image and its
	// views still live, and it needs no refresh. Reads the image without m_lock, as GetImage does.
	[[nodiscard]] bool          IsTextureCurrent(ImageId id, uint64_t generation) const noexcept;
	// The same for FindRenderTarget: the target is also GPU-owned already, so marking it written
	// changes nothing.
	[[nodiscard]] bool IsRenderTargetCurrent(ImageId id, uint64_t generation) const noexcept;
	void                        UpdateImage(ImageId id);
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);

	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear);
	void               InvalidateMemory(uint64_t address, uint64_t size);
	// What wrote guest memory through a buffer (for the KYTY_GPU_ZONES refresh log); shader
	// storage-buffer stores are the default.
	enum class GpuWriteSource : uint8_t { Fill, Copy, Shader };
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size,
	                                           GpuWriteSource source = GpuWriteSource::Shader);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t address, uint64_t size);
	// Queued publications remain authoritative until their guest backing writes complete.
	[[nodiscard]] bool HasPendingDownload(uint64_t address, uint64_t size);

	[[nodiscard]] bool IsMeta(uint64_t address);
	// Drain statistics only: native DCC metadata ranges seen by image lookups.
	[[nodiscard]] bool IsKnownDccMetadata(uint64_t address, uint64_t size);
	// A GPU write was recorded to guest memory. DCC slices checked on the GPU before it must be
	// checked again.
	void OnBufferGpuWrite(uint64_t address, uint64_t size);
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);

	void UnmapMemory(uint64_t address, uint64_t size);
	void ProcessDownloadImages();
	void RunGarbageCollector();

private:
	enum class TransferDirection { Upload, Download };
	struct TextureTransfer;
	struct ImageDownload;

	struct MetaDataInfo {
		enum class Type : uint8_t { CMask, FMask, HTile };

		Type     type;
		uint32_t clear_mask = UINT32_MAX;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
	};

	using ImageIds       = InlinePageOwnerList<ImageId, 16>;
	using ImagePageTable = MultiLevelPageTable<ImageIds, 20, 40, 10>;

	// Callers have validated the nonempty 40-bit range with TryGetPageRange.
	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		using FuncReturn = typename std::invoke_result<Func, uint64_t>::type;
		static constexpr bool RETURNS_BOOL = std::is_same_v<FuncReturn, bool>;
		const uint64_t page_end = (address + size - 1) >> ImagePageTable::kPageBits;
		for (uint64_t page = address >> ImagePageTable::kPageBits; page <= page_end; ++page) {
			if constexpr (RETURNS_BOOL) {
				if (func(page)) {
					break;
				}
			} else {
				func(page);
			}
		}
	}

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info);
	[[nodiscard]] ImageId     GetNullImage(const ImageDesc& desc);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id, std::vector<ImageId>* retired = nullptr);
	void                      FreeImage(ImageId id, std::vector<ImageId>* retired = nullptr);
	void                      TouchImage(Image& image);
	void                      TrackImage(ImageId id);
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id);
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	void                      TrackImageDownload(ImageId id, Image& image);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);
	[[nodiscard]] bool               SafeToDownload(const Image& image);

	// Caller holds m_lock; it also serializes the per-image query epoch.
	[[nodiscard]] ImageIds      FindImagesInRegion(uint64_t address, uint64_t size,
	                                               bool page_overlap) const;
	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	void                        RefreshImage(ImageId id);
	void                        MaterializeDccClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer);
	void                        FinishFind(ImageId id, const ImageDesc& desc,
	                                       uint32_t metadata_base_layer);
	// Applies clears found in GPU-written DCC metadata with conditional rendering. Returns false
	// when the target needs the CPU readback path.
	[[nodiscard]] bool MaterializeDccClearOnGpu(ImageId id, const ImageDesc& desc,
	                                            uint64_t slices_address, uint64_t slice_size,
	                                            uint32_t image_first, uint32_t count);
	[[nodiscard]] bool DccSlicesChecked(uint64_t address, uint64_t slice_size, uint32_t count,
	                                    uint32_t code_mask);
	void MarkDccSlicesChecked(uint64_t address, uint64_t slice_size, uint32_t count,
	                          uint32_t code_mask);
	void                        InitializeImage(ImageId id);
	[[nodiscard]] TextureTransfer
	BuildTextureTransfer(const Image& image, BindingType binding, TransferDirection direction) const;
	[[nodiscard]] ImageDownload BuildDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                       uint64_t destination_size, ImageDownload transfer);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	void CommitGpuWrite(Image& image);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	void ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                const vk::ImageSubresourceRange& range, const vk::ClearValue& clear);
	void PrepareImageCopy(Image& image);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;

	void               InvalidateCpuAliases(uint64_t address, uint64_t size);
	[[nodiscard]] bool DownloadImageMemory(ImageId id);
	[[nodiscard]] bool CollectGarbage(bool pressure_only);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingSpinLock                                  m_lock;
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	std::unordered_map<vk::Format, ImageId>           m_null_images;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	RangeSet                                          m_dcc_metadata_seen;
	struct DccCheckedSlice {
		uint64_t size      = 0;
		uint32_t code_mask = 0;
	};
	DccClearResolver                   m_dcc_resolver;
	bool                               m_dcc_gpu_clear  = false;
	uint64_t                           m_dcc_gpu_checks = 0;
	// Slice address -> slices whose GPU check still reflects the metadata. Any recorded GPU write
	// to the slice erases its entry; CPU writes leave the metadata CPU-dirty, which bypasses it.
	std::mutex                         m_dcc_checked_mutex;
	std::map<uint64_t, DccCheckedSlice> m_dcc_checked;
	std::mutex                                        m_pending_download_mutex;
	std::vector<GuestRange>                           m_pending_downloads;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	uint64_t         m_critical_gc_memory     = 3ull * 1024 * 1024 * 1024;
	uint64_t         m_gc_tick                = 0;
	uint64_t                                          m_last_pressure_gc_tick  = UINT64_MAX;
	mutable uint32_t m_image_query_epoch      = 0;
	bool             m_readback_linear_images = false;

	// FindImage results where the lookup found exactly one image with the same backing, keyed by
	// every request field SameBacking and the lookup's checks read. A lookup only sees registered
	// images, whose fields SameBacking reads never change, so a result stays right until an image
	// is registered or unregistered, which bumps m_image_set_generation. Caller holds m_lock.
	struct ImageLookup {
		uint64_t            generation = 0;
		GuestRange          data;
		vk::Extent3D        extent;
		ImageSubresources   resources;
		uint32_t            samples         = 0;
		uint32_t            bytes_per_block = 0;
		Prospero::TileMode  tile_mode       = Prospero::TileMode::kLinear;
		vk::Format          pixel_format    = vk::Format::eUndefined;
		Prospero::ImageType type            = Prospero::ImageType::kColor2D;
		bool                exact_format    = false;
		ImageId             id;

		[[nodiscard]] bool Matches(const ImageInfo& info, bool exact) const noexcept {
			return data == info.data && extent == info.extent && resources == info.resources &&
			       samples == info.samples && bytes_per_block == info.bytes_per_block &&
			       tile_mode == info.tile_mode && pixel_format == info.pixel_format &&
			       type == info.type && exact_format == exact;
		}
	};
	[[nodiscard]] static size_t ImageLookupSlot(const ImageInfo& info, bool exact) noexcept {
		auto hash = info.data.address ^ (info.data.size * 0x9e3779b97f4a7c15ull) ^
		            (static_cast<uint64_t>(info.pixel_format) << 1u) ^ (exact ? 1u : 0u);
		hash ^= hash >> 29u;
		hash *= 0xbf58476d1ce4e5b9ull;
		hash ^= hash >> 32u;
		return static_cast<size_t>(hash % ImageLookupCount);
	}
	static constexpr size_t              ImageLookupCount = 512;
	std::array<ImageLookup, ImageLookupCount> m_image_lookups {};
	// Each cache counts from its own base, so a generation remembered from one cache (a test's
	// earlier context, say) never matches another's.
	[[nodiscard]] static uint64_t        NextGenerationBase() noexcept {
		static std::atomic<uint64_t> caches {0};
		return (caches.fetch_add(1, std::memory_order_relaxed) + 1) << 40u;
	}
	// Changed under m_lock; IsTextureCurrent reads it without.
	std::atomic<uint64_t>                m_image_set_generation {NextGenerationBase()};
	// Bumped whenever an image becomes GPU-modified (MarkGpuModified); with the image set
	// generation it validates IsRegionGpuModified's per-thread record of clean pages.
	std::atomic<uint64_t>                m_gpu_modified_generation {NextGenerationBase()};
	void MarkGpuModified(Image& image) noexcept {
		if (!image.IsGpuModified()) {
			image.MarkGpuModified();
			m_gpu_modified_generation.fetch_add(1, std::memory_order_release);
		}
	}

	friend struct TextureCacheTestAccess;
	friend struct PerformanceMemoryTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
