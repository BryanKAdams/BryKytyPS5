#pragma once

#include "graphics/shader/recompiler/ir/Block.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

void ConstantPropagationPass(const BlockList& blocks);

// A pure per-lane operation: no memory, no side effects and no view of other lanes, so its value
// in a lane depends only on its operands in that lane.
bool IsLaneLocalOpcode(ValueOpcode opcode);

} // namespace Libs::Graphics::ShaderRecompiler::IR
