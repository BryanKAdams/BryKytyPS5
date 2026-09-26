#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// The guest skips an EXEC-masked region only when no lane is enabled; every enabled lane's
// writes are then masked by EXEC for free. The IR models that with a uniform branch on the whole
// mask and a select per write. This turns such a region into a real per-lane branch on the
// lane's EXEC bit, so the host GPU masks disabled lanes itself and they skip the region's
// arithmetic, loads and samples. Only regions whose results provably do not change are
// converted. Returns how many were.
uint32_t BranchExecRegions(Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR
