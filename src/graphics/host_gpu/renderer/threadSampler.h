#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_THREADSAMPLER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_THREADSAMPLER_H_

namespace Libs::Graphics {

// Debugging aid (Windows): with KYTY_DEBUG_SAMPLE_GPU=<period in microseconds> (values below 100
// select 500), a helper thread samples the calling thread's call stack and writes each 10-second
// window's stacks to sample-<name>-<window>.txt in the working directory, as module+offset
// frames (innermost first) for llvm-symbolizer. The sampled thread stops only while its
// registers and the top of its stack are copied. Call it from the thread to sample.
void StartThreadSampler(const char* name);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_THREADSAMPLER_H_
