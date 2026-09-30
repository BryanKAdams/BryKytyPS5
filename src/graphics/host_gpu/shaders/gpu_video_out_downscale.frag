#version 450

// Presents one mip level of a prepared frame to a smaller swapchain image (see PresentFilter).
// Each target pixel averages a two-texel box of the source on every downscaled axis, which
// cancels one-texel dither patterns at any phase; an axis that is not downscaled interpolates
// linearly like a blit.

layout(location = 0) out vec4 color;
layout(binding = 0) uniform sampler2D source; // linear, clamp to edge
layout(push_constant) uniform Downscale {
	ivec2 size;  // extent of the source level
	vec2  scale; // source texels per target pixel
	int   level;
};

// The box [c - 1, c + 1] (source texels) covers texels t, t + 1 and t + 2 with weights
// (1 - f) / 2, 1 / 2 and f / 2, where t + f = c - 1. One linear tap at a reads the first two,
// one at b the third. Positions are in texels, weights sum to one.
void Taps(float c, float s, out float a, out float wa, out float b, out float wb) {
	if (s > 1.0) {
		const float t = floor(c - 1.0);
		const float f = c - 1.0 - t;
		a             = t + 0.5 + 1.0 / (2.0 - f);
		wa            = 1.0 - 0.5 * f;
		b             = t + 2.5;
		wb            = 0.5 * f;
	} else {
		a  = c;
		wa = 1.0;
		b  = c;
		wb = 0.0;
	}
}

void main() {
	const vec2 c = gl_FragCoord.xy * scale;
	float      ax, wax, bx, wbx, ay, way, by, wby;
	Taps(c.x, scale.x, ax, wax, bx, wbx);
	Taps(c.y, scale.y, ay, way, by, wby);
	const vec2  texel = 1.0 / vec2(size);
	const float lod   = float(level);
	color = (textureLod(source, vec2(ax, ay) * texel, lod) * wax +
	         textureLod(source, vec2(bx, ay) * texel, lod) * wbx) *
	            way +
	        (textureLod(source, vec2(ax, by) * texel, lod) * wax +
	         textureLod(source, vec2(bx, by) * texel, lod) * wbx) *
	            wby;
}
