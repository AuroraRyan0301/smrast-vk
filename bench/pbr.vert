#version 450
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inColor;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    vec4 camPos;
    vec4 sunDir;
    vec4 pointPos;
    vec4 pointColor;
    vec4 timeParams;
} ubo;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outUV;
layout(location = 3) out vec4 outColor;

void main() {
    outWorldPos = inPos;
    outNormal   = inNormal;
    outUV       = inUV;
    outColor    = inColor;
    gl_Position = ubo.viewProj * vec4(inPos, 1.0);
}
