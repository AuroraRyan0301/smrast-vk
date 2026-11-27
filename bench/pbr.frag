#version 450
layout(location = 0) in vec3 inWorldPos;
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

layout(set = 0, binding = 1) uniform sampler2D albedoTex;

layout(location = 0) out vec4 outFragColor;

// ACES Filmic Tone Mapping
vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec3 N = normalize(inNormal);
    vec3 V = normalize(ubo.camPos.xyz - inWorldPos);
    vec3 L = normalize(ubo.sunDir.xyz);

    // Hardware-filtered texture sample (exercises H20 TMU + 1-cycle Quad Subgroup dFdx/dFdy!)
    vec4 tex = texture(albedoTex, inUV);
    vec3 baseColor = tex.rgb * inColor.rgb;
    float roughness = clamp(0.18 + 0.42 * (1.0 - tex.a), 0.08, 0.92);
    float metallic = (inColor.a < 0.95) ? 0.1 : 0.35;

    // Directional Sun Light (Cook-Torrance inspired GGX specular + Lambert diffuse)
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    float a2 = roughness * roughness;
    float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    float D = a2 / (3.14159265 * denom * denom + 1e-4);
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 sunSpec = (D * F) / (4.0 * NdotV * NdotL + 0.12);
    vec3 lo = (kD * baseColor / 3.14159265 + sunSpec) * vec3(2.8, 2.55, 2.25) * NdotL;

    // Dynamic Moving Point Light (cyan/orange accent)
    vec3 pVec = ubo.pointPos.xyz - inWorldPos;
    float pDist2 = max(dot(pVec, pVec), 0.5);
    vec3 pL = pVec * inversesqrt(pDist2);
    float pNdotL = max(dot(N, pL), 0.0);
    vec3 pH = normalize(pL + V);
    float pSpec = pow(max(dot(N, pH), 0.0), 48.0);
    float atten = 18.0 / (1.0 + 0.18 * pDist2);
    lo += (baseColor * 0.45 + vec3(0.7) * pSpec) * ubo.pointColor.rgb * (pNdotL * atten);

    // Hemisphere Sky/Ground Ambient + Fresnel Rim
    float hemi = N.y * 0.5 + 0.5;
    vec3 ambient = baseColor * mix(vec3(0.08, 0.09, 0.12), vec3(0.24, 0.34, 0.52), hemi);
    float fresnel = pow(1.0 - NdotV, 4.0);
    vec3 rim = vec3(0.22, 0.48, 0.88) * fresnel * 0.45;

    // Subtle distance fog for depth perception
    float dist = length(ubo.camPos.xyz - inWorldPos);
    float fog = 1.0 - exp(-dist * dist * 0.00055);
    vec3 fogCol = vec3(0.04, 0.06, 0.11);

    vec3 color = mix(ambient + lo + rim, fogCol, clamp(fog, 0.0, 0.78));
    color = pow(aces(color), vec3(1.0 / 2.2));
    outFragColor = vec4(color, inColor.a);
}
