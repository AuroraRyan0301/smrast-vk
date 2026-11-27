#version 450
layout(location = 0) out vec3 outColor;
void main() {
    int ix = gl_InstanceIndex & 15;
    int iy = gl_InstanceIndex >> 4;
    vec2 quad[6] = vec2[6](
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
    );
    vec2 cell = (vec2(float(ix), float(iy)) + quad[gl_VertexIndex] * 0.88 + 0.06) / 16.0;
    vec2 ndc = cell * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.25 + 0.5 * float((ix + iy) & 7) / 8.0, 1.0);
    outColor = vec3(float(ix) / 15.0, float(iy) / 15.0, 0.5 + 0.5 * float((ix ^ iy) & 1));
}
