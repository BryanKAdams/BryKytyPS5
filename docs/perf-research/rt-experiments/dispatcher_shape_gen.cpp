// Generates GLSL compute shaders that put comparable block bodies in different control-flow shapes:
//   D: recompiler-style dispatcher (loop + switch on a per-lane pc)
//   U: dispatcher whose pc is made wave-uniform with subgroupBroadcastFirst
//   S: structured (a loop around a chain of if/else diamonds)
//   L: structured, short live ranges (each block reads only the previous block's values)
//   E: the same blocks as L inside a dispatcher
// D, U and S keep every value live until the end; L and E model SSA values that cross one block.
// usage: dispatcher_shape_gen <D|U|S|L|E> <blocks> <vars> <seed>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>

static uint32_t rng = 1;
static uint32_t Next() {
	rng ^= rng << 13;
	rng ^= rng >> 17;
	rng ^= rng << 5;
	return rng;
}

static void Body(uint32_t vars) {
	for (int s = 0; s < 6; s++) {
		const uint32_t a = Next() % vars, b = Next() % vars, c = Next() % vars, d = Next() % vars;
		if (s % 2 == 0) {
			std::printf("      v%u = v%u * v%u + v%u;\n", a, b, c, d);
		} else {
			std::printf("      v%u ^= (v%u >> %u);\n", a, b, 1 + Next() % 7);
		}
	}
}

int main(int argc, char** argv) {
	if (argc != 5) {
		std::fprintf(stderr, "usage: dispatcher_shape_gen <D|U|S|L|E> <blocks> <vars> <seed>\n");
		return 1;
	}
	const char     mode   = argv[1][0];
	const uint32_t blocks = static_cast<uint32_t>(std::atoi(argv[2]));
	const uint32_t vars   = static_cast<uint32_t>(std::atoi(argv[3]));
	const uint32_t seed   = static_cast<uint32_t>(std::atoi(argv[4]));
	rng                   = 0x9e3779b9u ^ (blocks * 7919u) ^ vars;
	std::printf("#version 460\n#extension GL_KHR_shader_subgroup_ballot : require\n");
	std::printf("layout(local_size_x = 64) in;\n");
	std::printf("layout(std430, set = 0, binding = 0) buffer Buf { uint data[]; };\n");
	std::printf("void main() {\n  uint gid = gl_GlobalInvocationID.x;\n");
	if (mode == 'L' || mode == 'E') {
		// Short live ranges: block b defines vars [b*k, b*k+k) from the previous block's vars,
		// like SSA values that cross only one or two guest blocks. L runs the blocks in
		// structured order; E puts the same blocks in a recompiler-style dispatcher.
		const uint32_t k = vars / blocks;
		for (uint32_t v = 0; v < vars; v++) {
			std::printf("  uint v%u = data[(gid + %uu) & 4095u] ^ %uu;\n", v, v * 13u, seed + v);
		}
		const auto block = [&](uint32_t b) {
			for (uint32_t i = 0; i < k; i++) {
				const uint32_t prev = b == 0 ? 0 : (b - 1) * k;
				std::printf("      v%u = v%u * v%u + (v%u >> %u);\n", b * k + i, prev + Next() % k,
				            prev + Next() % k, prev + Next() % k, 1 + Next() % 7);
			}
		};
		if (mode == 'L') {
			for (uint32_t b = 0; b < blocks; b++) {
				std::printf("    {\n");
				block(b);
				std::printf("    }\n    if ((v%u & 1u) != 0u) { v%u ^= 0x55u; }\n", b * k,
				            b * k + 1);
			}
		} else {
			// z is 0 in practice but unknown to the compiler, so the case order stays opaque.
			std::printf("  uint z = data[4095] >> 31;\n  uint pc = 0u;\n");
			std::printf("  while (pc != 0xffffffffu) {\n    switch (pc) {\n");
			for (uint32_t b = 0; b < blocks; b++) {
				std::printf("    case %uu: {\n", b);
				block(b);
				std::printf("      if ((v%u & 1u) != 0u) { v%u ^= 0x55u; }\n", b * k, b * k + 1);
				std::printf("      pc = %s;\n      break;\n    }\n",
				            b + 1 < blocks ? (std::to_string(b + 1) + "u + z").c_str() : "0xffffffffu");
			}
			std::printf("    default: pc = 0xffffffffu; break;\n    }\n  }\n");
		}
		std::printf("  uint acc = 0u;\n");
		for (uint32_t v = (blocks - 1) * k; v < vars; v++) {
			std::printf("  acc ^= v%u;\n", v);
		}
		std::printf("  data[gid] = acc;\n}\n");
		return 0;
	}
	for (uint32_t v = 0; v < vars; v++) {
		std::printf("  uint v%u = data[(gid + %uu) & 4095u] ^ %uu;\n", v, v * 13u, seed + v);
	}
	if (mode == 'S') {
		std::printf("  for (uint iter = 0u; iter < (data[0] & 15u); iter++) {\n");
		for (uint32_t b = 0; b + 1 < blocks; b += 2) {
			std::printf("    if ((v%u & 1u) != 0u) {\n", Next() % vars);
			Body(vars);
			std::printf("    } else {\n");
			Body(vars);
			std::printf("    }\n");
		}
		std::printf("  }\n");
	} else {
		// Block 0 starts from a per-lane value, as a guest pc selected by per-lane data would.
		std::printf("  uint pc = %s;\n  uint iter = 0u;\n",
		            mode == 'U' ? "subgroupBroadcastFirst(v0 & 1u)" : "v0 & 1u");
		std::printf("  while (pc != 0xffffffffu) {\n    switch (pc) {\n");
		for (uint32_t b = 0; b < blocks; b++) {
			std::printf("    case %uu: {\n", b);
			Body(vars);
			uint32_t t = b + 1;
			uint32_t f = b + 2 + Next() % 5;
			if (b % 16 == 15 && b >= 8) {
				f = b - 8; // a back edge, like a guest loop
			}
			const std::string tt = t < blocks ? std::to_string(t) + "u" : "0xffffffffu";
			const std::string ff = f < blocks ? std::to_string(f) + "u" : "0xffffffffu";
			std::printf("      pc = (v%u & 1u) != 0u ? %s : %s;\n      break;\n    }\n",
			            Next() % vars, tt.c_str(), ff.c_str());
		}
		std::printf("    default: pc = 0xffffffffu; break;\n    }\n");
		std::printf("    iter++;\n    if (iter > 4096u) pc = 0xffffffffu;\n");
		if (mode == 'U') {
			std::printf("    pc = subgroupBroadcastFirst(pc);\n");
		}
		std::printf("  }\n");
	}
	std::printf("  uint acc = 0u;\n");
	for (uint32_t v = 0; v < vars; v++) {
		std::printf("  acc ^= v%u;\n", v);
	}
	std::printf("  data[gid] = acc;\n}\n");
	return 0;
}
