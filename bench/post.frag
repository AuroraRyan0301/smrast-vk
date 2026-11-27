#version 450
layout(location = 0) in vec2 inUV;
layout(set = 0, binding = 0) uniform sampler2D rttTex;
layout(location = 0) out vec4 outFrag;
void main() {
    vec3 c = texture(rttTex, inUV).rgb;
    // Vignette + slight saturation boost to prove Pass 2 FS ran over Pass 1 RTT!
    float v = 1.0 - 0.2 * length(inUV - 0.5);
    outFrag = vec4(c * v, 1.0);
}
