#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace Libs::Graphics {

struct ShaderStageRuntime;

struct TextureBinding {
	ImageId                    image_id;
	vk::ImageView              image_view = nullptr;
	TextureCache::ImageDesc    desc;
	vk::ImageLayout            layout = vk::ImageLayout::eUndefined;
	std::vector<vk::ImageView> mip_views;
};

struct PreparedBindings {
	struct BufferSource {
		uint64_t address = 0;
		uint64_t size    = 0;
		BufferId id;
	};
	// A view FindTexture returned for a binding of `image` with `info`, and the image set's
	// generation then (see TextureCache::IsTextureCurrent); generation 0: none.
	struct ViewMemo {
		vk::ImageView view;
		ImageId       image;
		ImageViewInfo info;
		uint64_t      generation = 0;
	};
	// What an image binding was resolved from, so that the next draw binding the same descriptor
	// can keep it (see TextureCache::RefindImage).
	struct ImageSource {
		ShaderRecompiler::IR::ImageResource   resource;
		ShaderRecompiler::IR::DescriptorValue value;
		ImageId                               found;          // FindImage's result.
		uint64_t                              generation = 0; // 0: not reusable.
		// A null descriptor's binding: the texture cache's null image for the resource's format,
		// which lives as long as the cache and has no guest memory to refresh.
		bool                                  null_image = false;
		uint32_t                              metadata_base_layer = 0;
		// The view FindTexture last returned for a sampled binding of view_image, and the image
		// set's generation then (see TextureCache::IsTextureCurrent); generation 0: none.
		vk::ImageView                         view;
		ImageId                               view_image;
		ImageViewInfo                         view_info;
		uint64_t                              view_generation = 0;
		// The same memo kept with the texture description that resolved this binding, so another
		// slot or draw resolving that description starts from the view (see ResolveTextureInto).
		// The entry may describe another texture by now: a memo is only used for its own image
		// and view, and IsTextureCurrent checks it as the slot's own.
		ViewMemo*                             view_memo = nullptr;
	};

	// The draw owns the immutable compiled-program/runtime-snapshot association through commit.
	const ShaderStageRuntime* runtime = nullptr;
	// Keep the resolved guest range through cache preparation; only the host buffer ID may
	// become stale and need resolving again when bindings are rebound.
	std::vector<BufferSource>             buffer_sources;
	std::vector<vk::DescriptorBufferInfo> buffers;
	std::vector<TextureBinding>           images;
	std::vector<ImageSource>              image_sources;
	std::vector<vk::Sampler>              samplers;
	vk::DescriptorBufferInfo              gds {nullptr, 0, VK_WHOLE_SIZE};
	vk::DescriptorBufferInfo              flattened_srt;
	vk::DescriptorBufferInfo              shader_data_buffer;
	std::vector<uint32_t>                 shader_data;
	// KYTY_DEBUG_DRAW_PHASES: the previous draw's program and descriptor hashes (see
	// NoteBindingRepeat).
	// The stage's images as its last draw left them, for a draw with the same program and image
	// descriptors (see RenderExecutor::ReuseImageGroup): the bindings and views stand while no
	// lookup could find another image (TextureCache::CurrentTargetState) and each image stays
	// clean. program nullptr: no group (a storage image, or a view that is not kept).
	struct ImageGroup {
		const void*                                        program = nullptr;
		std::vector<ShaderRecompiler::IR::DescriptorValue> images;
		std::vector<uint32_t>                              mip_counters;
		TextureCache::TargetState                          state {};
	} image_group;
	// This draw kept the group so far: PrepareBindings resolved nothing, and RebindImages keeps
	// the views if the group still stands then.
	bool image_group_kept = false;

	struct RepeatKey {
		const void* program  = nullptr;
		uint64_t    images   = 0;
		uint64_t    buffers  = 0;
		uint64_t    samplers = 0;
	} repeat_key;
};

[[nodiscard]] vk::DescriptorType
NativeDescriptorType(ShaderRecompiler::IR::DescriptorBindingKind kind);
[[nodiscard]] uint32_t
NativeDescriptorCount(const ShaderRecompiler::IR::DescriptorBinding& binding);
[[nodiscard]] vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture,
                                                    uint32_t              element = 0);

template <typename T>
[[nodiscard]] T DecodeNativeDescriptor(const ShaderRecompiler::IR::DescriptorValue& value) {
	static_assert(std::is_trivially_copyable_v<T>);
	static_assert(sizeof(T) % sizeof(uint32_t) == 0);
	T result {};
	EXIT_IF(value.dword_count < sizeof(result) / sizeof(uint32_t));
	std::memcpy(&result, value.dwords.data(), sizeof(result));
	return result;
}

[[nodiscard]] bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor,
                                                   bool r128 = false);
void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
