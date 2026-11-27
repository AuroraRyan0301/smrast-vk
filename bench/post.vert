#version 450
layout(location = 0) out vec2 outUV;
void main() {
    vec2 pos[4] = vec2[4](vec2(-1,-1), vec2(1,-1), vec2(-1,1), vec2(1,1));
    vec2 uv[4]  = vec2[4](vec2(0,0),   vec2(1,0),  vec2(0,1),  vec2(1,1));
    gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
    outUV = uv[gl_VertexIndex];
}
