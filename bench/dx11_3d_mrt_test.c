// dx11_3d_mrt_test.c — Full 2-Pass DirectX 11 Deferred PBR Renderer (MRT + HDR RGBA16F + R11G11B10F + DrawIndexedInstancedIndirect)
// Runs unmodified Windows HLSL (vs_5_0 + ps_5_0) -> DXBC -> Unmodified DXVK 2.4.1 -> SmRast-VK v0.3 -> NVIDIA H20 78 SMs!
#define COBJMACROS
#define INITGUID
#include <windows.h>
#include <initguid.h>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

typedef HRESULT (WINAPI *PFN_D3DCompile)(
    LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*,
    LPCSTR, LPCSTR, UINT, UINT, ID3DBlob**, ID3DBlob**);

// PASS 1: 3D Geometry -> 3-RT HDR Deferred G-Buffer (RT0=RGBA8 Albedo, RT1=RGBA16F WorldNormal, RT2=R11G11B10F Emissive/Spec)
static const char* HLSL_GBUFFER =
"cbuffer SceneCB : register(b0) {\n"
"    row_major float4x4 mvp;\n"
"    row_major float4x4 model;\n"
"};\n"
"Texture2D albedoTex : register(t0);\n"
"SamplerState linearSamp : register(s0);\n"
"struct VSIn { float3 pos : POSITION; float3 nrm : NORMAL; float2 uv : TEXCOORD0; float4 col : COLOR; };\n"
"struct PSIn { float4 pos : SV_POSITION; float3 wpos : TEXCOORD0; float3 wnrm : TEXCOORD1; float2 uv : TEXCOORD2; float4 col : COLOR; };\n"
"struct GOut {\n"
"    float4 rt0 : SV_TARGET0; // RGBA8 Albedo\n"
"    float4 rt1 : SV_TARGET1; // RGBA16F World Normal + Roughness\n"
"    float4 rt2 : SV_TARGET2; // R11G11B10F HDR Emissive / Rim Tint\n"
"};\n"
"PSIn GBufferVS(VSIn i) {\n"
"    PSIn o;\n"
"    float4 wp = mul(float4(i.pos, 1.0), model);\n"
"    o.pos  = mul(float4(i.pos, 1.0), mvp);\n"
"    o.wpos = wp.xyz;\n"
"    o.wnrm = normalize(mul(float4(i.nrm, 0.0), model).xyz);\n"
"    o.uv   = i.uv;\n"
"    o.col  = i.col;\n"
"    return o;\n"
"}\n"
"GOut GBufferPS(PSIn i) {\n"
"    GOut o;\n"
"    float4 t = albedoTex.Sample(linearSamp, i.uv);\n"
"    o.rt0 = float4(t.rgb * i.col.rgb, 1.0);\n"
"    o.rt1 = float4(normalize(i.wnrm) * 0.5 + 0.5, 0.25);\n"
"    // Encode bright HDR emissive bands > 1.0 to verify R11G11B10F HDR headroom!\n"
"    float band = (frac(i.uv.x * 4.0) < 0.14 || frac(i.uv.y * 4.0) < 0.14) ? 1.8 : 0.05;\n"
"    o.rt2 = float4(i.col.rgb * band, 1.0);\n"
"    return o;\n"
"}\n";

// PASS 2: Fullscreen Deferred Lighting Resolve (samples all 3 G-Buffer RTs via Hardware H20 TMUs!)
static const char* HLSL_LIGHTING =
"Texture2D gAlbedo   : register(t0);\n"
"Texture2D gNormal   : register(t1);\n"
"Texture2D gEmissive : register(t2);\n"
"SamplerState ptSamp : register(s0);\n"
"struct VSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };\n"
"VSOut FullVS(uint vid : SV_VERTEXID) {\n"
"    VSOut o;\n"
"    float2 uv = float2((vid << 1) & 2, vid & 2);\n"
"    o.pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);\n"
"    o.uv  = uv;\n"
"    return o;\n"
"}\n"
"float4 LightPS(VSOut i) : SV_TARGET {\n"
"    float4 alb = gAlbedo.Sample(ptSamp, i.uv);\n"
"    float4 nrmRaw = gNormal.Sample(ptSamp, i.uv);\n"
"    float3 emi = gEmissive.Sample(ptSamp, i.uv).rgb;\n"
"    if (alb.a < 0.5) {\n"
"        // Deep space studio gradient background\n"
"        return float4(lerp(float3(0.03, 0.05, 0.12), float3(0.09, 0.14, 0.24), i.uv.y), 1.0);\n"
"    }\n"
"    float3 N = normalize(nrmRaw.xyz * 2.0 - 1.0);\n"
"    float3 L1 = normalize(float3(0.6, 0.8, 0.5));\n"
"    float3 L2 = normalize(float3(-0.7, 0.3, 0.6));\n"
"    float3 V  = normalize(float3(0.0, 0.4, 1.0));\n"
"    float diff1 = max(dot(N, L1), 0.0);\n"
"    float diff2 = max(dot(N, L2), 0.0);\n"
"    float spec1 = pow(max(dot(N, normalize(L1 + V)), 0.0), 48.0);\n"
"    float rim   = pow(1.0 - max(dot(N, V), 0.0), 3.0);\n"
"    float3 hdr = alb.rgb * (0.14 + diff1 * float3(1.15, 1.05, 0.95) + diff2 * float3(0.25, 0.55, 0.95))\n"
"               + float3(1.0, 0.95, 0.9) * spec1 * 0.85\n"
"               + float3(0.2, 0.6, 1.0) * rim * 0.35\n"
"               + emi * 0.42;\n"
"    // ACES Filmic Tone Mapping + Gamma\n"
"    float3 ldr = saturate((hdr * (2.51 * hdr + 0.03)) / (hdr * (2.43 * hdr + 0.59) + 0.14));\n"
"    return float4(pow(ldr, 1.0 / 2.2), 1.0);\n"
"}\n";

typedef struct { float px, py, pz, nx, ny, nz, u, v, r, g, b, a; } Vtx;

static void mat4_mul(const float a[16], const float b[16], float o[16]) {
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            o[r*4+c] = a[r*4+0]*b[0*4+c] + a[r*4+1]*b[1*4+c] + a[r*4+2]*b[2*4+c] + a[r*4+3]*b[3*4+c];
}
static void mat4_rot_xy(float ax, float ay, float m[16]) {
    float cx = cosf(ax), sx = sinf(ax), cy = cosf(ay), sy = sinf(ay);
    // row-major rotation Y * X
    float rx[16] = {1,0,0,0, 0,cx,sx,0, 0,-sx,cx,0, 0,0,0,1};
    float ry[16] = {cy,0,-sy,0, 0,1,0,0, sy,0,cy,0, 0,0,0,1};
    mat4_mul(rx, ry, m);
}
static void mat4_persp_lh(float fov, float asp, float zn, float zf, float m[16]) {
    memset(m, 0, 64);
    float y = 1.0f / tanf(fov * 0.5f), x = y / asp;
    m[0] = x; m[5] = y; m[10] = zf / (zf - zn); m[11] = 1.0f; m[14] = -zn * zf / (zf - zn);
}
static void mat4_trans(float tx, float ty, float tz, float m[16]) {
    memset(m, 0, 64);
    m[0]=m[5]=m[10]=m[15]=1.0f; m[12]=tx; m[13]=ty; m[14]=tz;
}

static void add_cube(Vtx* v, uint32_t* idx, int* nv, int* ni) {
    struct Face { float n[3]; float p[4][3]; float c[3]; } faces[6] = {
        {{0,0,-1}, {{-1,-1,-1},{-1,1,-1},{1,1,-1},{1,-1,-1}}, {0.20f, 0.85f, 0.98f}},
        {{0,0, 1}, {{ 1,-1, 1},{ 1,1, 1},{-1,1, 1},{-1,-1, 1}}, {0.98f, 0.45f, 0.22f}},
        {{1,0, 0}, {{ 1,-1,-1},{ 1,1,-1},{ 1,1, 1},{ 1,-1, 1}}, {0.95f, 0.78f, 0.22f}},
        {{-1,0,0}, {{-1,-1, 1},{-1,1, 1},{-1,1,-1},{-1,-1,-1}}, {0.65f, 0.38f, 0.98f}},
        {{0,1, 0}, {{-1, 1,-1},{-1,1, 1},{ 1,1, 1},{ 1,1,-1}}, {0.32f, 0.92f, 0.55f}},
        {{0,-1,0}, {{-1,-1, 1},{-1,-1,-1},{ 1,-1,-1},{ 1,-1, 1}}, {0.92f, 0.30f, 0.42f}},
    };
    const float uvs[4][2] = {{0,1},{0,0},{1,0},{1,1}};
    for (int f = 0; f < 6; ++f) {
        int b = *nv;
        for (int k = 0; k < 4; ++k) {
            Vtx* dst = &v[(*nv)++];
            dst->px = faces[f].p[k][0]; dst->py = faces[f].p[k][1]; dst->pz = faces[f].p[k][2];
            dst->nx = faces[f].n[0];    dst->ny = faces[f].n[1];    dst->nz = faces[f].n[2];
            dst->u  = uvs[k][0];        dst->v  = uvs[k][1];
            dst->r  = faces[f].c[0];    dst->g  = faces[f].c[1];    dst->b  = faces[f].c[2]; dst->a = 1.0f;
        }
        idx[(*ni)++] = b+0; idx[(*ni)++] = b+1; idx[(*ni)++] = b+2;
        idx[(*ni)++] = b+0; idx[(*ni)++] = b+2; idx[(*ni)++] = b+3;
    }
}

int main(void) {
    FILE* logf = fopen("Z:\\tmp\\dx11_mrt_result.txt", "w");
    if (!logf) logf = stdout;

    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0, fl_out = 0;
    ID3D11Device* dev = NULL; ID3D11DeviceContext* ctx = NULL;
    HRESULT hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, 0, &fl, 1,
                                   D3D11_SDK_VERSION, &dev, &fl_out, &ctx);
    fprintf(logf, "D3D11CreateDevice hr=0x%08lx fl=0x%x\n", (unsigned long)hr, (unsigned)fl_out); fflush(logf);
    if (FAILED(hr)) return 1;

    HMODULE d3dc = LoadLibraryA("d3dcompiler_47.dll");
    PFN_D3DCompile pCompile = (PFN_D3DCompile)GetProcAddress(d3dc, "D3DCompile");
    ID3DBlob *gvsb=NULL, *gpsb=NULL, *lvsb=NULL, *lpsb=NULL, *errb=NULL;
    pCompile(HLSL_GBUFFER, strlen(HLSL_GBUFFER), "gb.hlsl", NULL, NULL, "GBufferVS", "vs_5_0", 0, 0, &gvsb, &errb);
    pCompile(HLSL_GBUFFER, strlen(HLSL_GBUFFER), "gb.hlsl", NULL, NULL, "GBufferPS", "ps_5_0", 0, 0, &gpsb, &errb);
    pCompile(HLSL_LIGHTING, strlen(HLSL_LIGHTING), "lt.hlsl", NULL, NULL, "FullVS", "vs_5_0", 0, 0, &lvsb, &errb);
    pCompile(HLSL_LIGHTING, strlen(HLSL_LIGHTING), "lt.hlsl", NULL, NULL, "LightPS", "ps_5_0", 0, 0, &lpsb, &errb);
    if (!gvsb || !gpsb || !lvsb || !lpsb) {
        fprintf(logf, "HLSL compile failed: %s\n", errb ? (char*)ID3D10Blob_GetBufferPointer(errb) : "null");
        return 2;
    }

    ID3D11VertexShader *gvs=NULL, *lvs=NULL; ID3D11PixelShader *gps=NULL, *lps=NULL;
    ID3D11Device_CreateVertexShader(dev, ID3D10Blob_GetBufferPointer(gvsb), ID3D10Blob_GetBufferSize(gvsb), NULL, &gvs);
    ID3D11Device_CreatePixelShader(dev,  ID3D10Blob_GetBufferPointer(gpsb), ID3D10Blob_GetBufferSize(gpsb), NULL, &gps);
    ID3D11Device_CreateVertexShader(dev, ID3D10Blob_GetBufferPointer(lvsb), ID3D10Blob_GetBufferSize(lvsb), NULL, &lvs);
    ID3D11Device_CreatePixelShader(dev,  ID3D10Blob_GetBufferPointer(lpsb), ID3D10Blob_GetBufferSize(lpsb), NULL, &lps);

    D3D11_INPUT_ELEMENT_DESC ied[4] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 32, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    ID3D11InputLayout* il = NULL;
    ID3D11Device_CreateInputLayout(dev, ied, 4, ID3D10Blob_GetBufferPointer(gvsb), ID3D10Blob_GetBufferSize(gvsb), &il);

    const UINT W = 512, H = 512;
    DXGI_FORMAT rt_fmts[3] = { DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R11G11B10_FLOAT };
    ID3D11Texture2D* gtex[3] = {0}; ID3D11RenderTargetView* grtv[3] = {0}; ID3D11ShaderResourceView* gsrv[3] = {0};
    for (int i = 0; i < 3; ++i) {
        D3D11_TEXTURE2D_DESC td; memset(&td, 0, sizeof(td));
        td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1; td.Format = rt_fmts[i];
        td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        ID3D11Device_CreateTexture2D(dev, &td, NULL, &gtex[i]);
        ID3D11Device_CreateRenderTargetView(dev, (ID3D11Resource*)gtex[i], NULL, &grtv[i]);
        ID3D11Device_CreateShaderResourceView(dev, (ID3D11Resource*)gtex[i], NULL, &gsrv[i]);
    }
    D3D11_TEXTURE2D_DESC dtd; memset(&dtd, 0, sizeof(dtd));
    dtd.Width = W; dtd.Height = H; dtd.MipLevels = 1; dtd.ArraySize = 1; dtd.Format = DXGI_FORMAT_D32_FLOAT;
    dtd.SampleDesc.Count = 1; dtd.Usage = D3D11_USAGE_DEFAULT; dtd.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    ID3D11Texture2D* dtex = NULL; ID3D11DepthStencilView* dsv = NULL;
    ID3D11Device_CreateTexture2D(dev, &dtd, NULL, &dtex);
    ID3D11Device_CreateDepthStencilView(dev, (ID3D11Resource*)dtex, NULL, &dsv);

    D3D11_TEXTURE2D_DESC ftd; memset(&ftd, 0, sizeof(ftd));
    ftd.Width = W; ftd.Height = H; ftd.MipLevels = 1; ftd.ArraySize = 1; ftd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    ftd.SampleDesc.Count = 1; ftd.Usage = D3D11_USAGE_DEFAULT; ftd.BindFlags = D3D11_BIND_RENDER_TARGET;
    ID3D11Texture2D* final_tex = NULL; ID3D11RenderTargetView* final_rtv = NULL;
    ID3D11Device_CreateTexture2D(dev, &ftd, NULL, &final_tex);
    ID3D11Device_CreateRenderTargetView(dev, (ID3D11Resource*)final_tex, NULL, &final_rtv);

    // Procedural 256x256 Checkerboard Albedo Texture (separate Texture2D + SamplerState!)
    uint8_t* tdata = (uint8_t*)malloc(256 * 256 * 4);
    for (int y = 0; y < 256; ++y) for (int x = 0; x < 256; ++x) {
        int c = (((x >> 4) ^ (y >> 4)) & 1) ? 235 : 135;
        int o = (y * 256 + x) * 4;
        tdata[o+0] = c; tdata[o+1] = c; tdata[o+2] = c; tdata[o+3] = 255;
    }
    D3D11_TEXTURE2D_DESC atd; memset(&atd, 0, sizeof(atd));
    atd.Width = 256; atd.Height = 256; atd.MipLevels = 1; atd.ArraySize = 1; atd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    atd.SampleDesc.Count = 1; atd.Usage = D3D11_USAGE_IMMUTABLE; atd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA asd = {tdata, 256 * 4, 0};
    ID3D11Texture2D* atex = NULL; ID3D11ShaderResourceView* asrv = NULL;
    ID3D11Device_CreateTexture2D(dev, &atd, &asd, &atex);
    ID3D11Device_CreateShaderResourceView(dev, (ID3D11Resource*)atex, NULL, &asrv);
    free(tdata);

    D3D11_SAMPLER_DESC sd; memset(&sd, 0, sizeof(sd));
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    ID3D11SamplerState* samp = NULL;
    ID3D11Device_CreateSamplerState(dev, &sd, &samp);

    // Geometry: 3D Cube + Indirect Draw Args Buffer!
    Vtx verts[64]; uint32_t idxs[64]; int nv = 0, ni = 0;
    add_cube(verts, idxs, &nv, &ni);
    D3D11_BUFFER_DESC vbd; memset(&vbd, 0, sizeof(vbd));
    vbd.ByteWidth = nv * sizeof(Vtx); vbd.Usage = D3D11_USAGE_IMMUTABLE; vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA vsd = {verts, 0, 0};
    ID3D11Buffer* vb = NULL; ID3D11Device_CreateBuffer(dev, &vbd, &vsd, &vb);

    D3D11_BUFFER_DESC ibd; memset(&ibd, 0, sizeof(ibd));
    ibd.ByteWidth = ni * sizeof(uint32_t); ibd.Usage = D3D11_USAGE_IMMUTABLE; ibd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    D3D11_SUBRESOURCE_DATA isd = {idxs, 0, 0};
    ID3D11Buffer* ib = NULL; ID3D11Device_CreateBuffer(dev, &ibd, &isd, &ib);

    // GPU-Driven DrawIndexedInstancedIndirect Arguments Buffer!
    uint32_t ind_args[5] = {(uint32_t)ni, 1u, 0u, 0u, 0u};
    D3D11_BUFFER_DESC ind_bd; memset(&ind_bd, 0, sizeof(ind_bd));
    ind_bd.ByteWidth = sizeof(ind_args); ind_bd.Usage = D3D11_USAGE_DEFAULT;
    ind_bd.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
    D3D11_SUBRESOURCE_DATA ind_sd = {ind_args, 0, 0};
    ID3D11Buffer* ind_buf = NULL; ID3D11Device_CreateBuffer(dev, &ind_bd, &ind_sd, &ind_buf);

    // Constant Buffer (MVP + Model matrices)
    struct CBData { float mvp[16]; float model[16]; } cbdata;
    float rot[16], trans[16], mv[16], proj[16];
    mat4_rot_xy(0.55f, 0.72f, rot);
    mat4_trans(0.0f, 0.0f, 3.8f, trans);
    mat4_mul(rot, trans, mv);
    mat4_persp_lh(0.95f, 1.0f, 0.1f, 50.0f, proj);
    mat4_mul(mv, proj, cbdata.mvp);
    memcpy(cbdata.model, rot, 64);

    D3D11_BUFFER_DESC cbd; memset(&cbd, 0, sizeof(cbd));
    cbd.ByteWidth = sizeof(cbdata); cbd.Usage = D3D11_USAGE_IMMUTABLE; cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA csd = {&cbdata, 0, 0};
    ID3D11Buffer* cb = NULL; ID3D11Device_CreateBuffer(dev, &cbd, &csd, &cb);

    // --- PASS 1: 3-RT HDR G-BUFFER + DEPTH + GPU-DRIVEN DrawIndexedInstancedIndirect ---
    float z4[4] = {0,0,0,0};
    for (int i = 0; i < 3; ++i) ID3D11DeviceContext_ClearRenderTargetView(ctx, grtv[i], z4);
    ID3D11DeviceContext_ClearDepthStencilView(ctx, dsv, D3D11_CLEAR_DEPTH, 1.0f, 0);

    D3D11_VIEWPORT vp = {0, 0, (float)W, (float)H, 0.0f, 1.0f};
    UINT stride = sizeof(Vtx), off = 0;
    ID3D11DeviceContext_OMSetRenderTargets(ctx, 3, grtv, dsv);
    ID3D11DeviceContext_RSSetViewports(ctx, 1, &vp);
    ID3D11DeviceContext_IASetInputLayout(ctx, il);
    ID3D11DeviceContext_IASetVertexBuffers(ctx, 0, 1, &vb, &stride, &off);
    ID3D11DeviceContext_IASetIndexBuffer(ctx, ib, DXGI_FORMAT_R32_UINT, 0);
    ID3D11DeviceContext_IASetPrimitiveTopology(ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D11DeviceContext_VSSetShader(ctx, gvs, NULL, 0);
    ID3D11DeviceContext_VSSetConstantBuffers(ctx, 0, 1, &cb);
    ID3D11DeviceContext_PSSetShader(ctx, gps, NULL, 0);
    ID3D11DeviceContext_PSSetShaderResources(ctx, 0, 1, &asrv);
    ID3D11DeviceContext_PSSetSamplers(ctx, 0, 1, &samp);
    ID3D11DeviceContext_DrawIndexedInstancedIndirect(ctx, ind_buf, 0);
    fprintf(logf, "Pass 1 (3-RT HDR G-Buffer + DrawIndexedInstancedIndirect) recorded!\n"); fflush(logf);

    // --- PASS 2: DEFERRED LIGHTING RESOLVE SAMPLING ALL 3 G-BUFFER RTs ---
    ID3D11RenderTargetView* null_rtvs[3] = {NULL, NULL, NULL};
    ID3D11DeviceContext_OMSetRenderTargets(ctx, 3, null_rtvs, NULL);
    ID3D11DeviceContext_OMSetRenderTargets(ctx, 1, &final_rtv, NULL);
    ID3D11DeviceContext_IASetInputLayout(ctx, NULL);
    ID3D11DeviceContext_VSSetShader(ctx, lvs, NULL, 0);
    ID3D11DeviceContext_PSSetShader(ctx, lps, NULL, 0);
    ID3D11DeviceContext_PSSetShaderResources(ctx, 0, 3, gsrv);
    ID3D11DeviceContext_PSSetSamplers(ctx, 0, 1, &samp);
    ID3D11DeviceContext_Draw(ctx, 3, 0);
    fprintf(logf, "Pass 2 (Deferred PBR Lighting Resolve sampling RGBA8+RGBA16F+R11G11B10F) recorded!\n"); fflush(logf);

    // Readback final lit 3D deferred frame
    ftd.Usage = D3D11_USAGE_STAGING; ftd.BindFlags = 0; ftd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Texture2D* stg = NULL; ID3D11Device_CreateTexture2D(dev, &ftd, NULL, &stg);
    ID3D11DeviceContext_CopyResource(ctx, (ID3D11Resource*)stg, (ID3D11Resource*)final_tex);
    ID3D11DeviceContext_Flush(ctx);

    D3D11_MAPPED_SUBRESOURCE mapped; memset(&mapped, 0, sizeof(mapped));
    hr = ID3D11DeviceContext_Map(ctx, (ID3D11Resource*)stg, 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr) && mapped.pData) {
        const uint8_t* base = (const uint8_t*)mapped.pData;
        const uint8_t* p_bg = base + 8 * mapped.RowPitch + 8 * 4;
        const uint8_t* p_ct = base + (H/2) * mapped.RowPitch + (W/2) * 4;
        fprintf(logf, "DEFERRED_MRT_PIXEL BG=(%u,%u,%u) CENTER=(%u,%u,%u)\n",
                p_bg[0], p_bg[1], p_bg[2], p_ct[0], p_ct[1], p_ct[2]);
        FILE* ppm = fopen("Z:\\tmp\\dx11_deferred_mrt_h20.ppm", "wb");
        if (ppm) {
            fprintf(ppm, "P6\n%u %u\n255\n", W, H);
            for (UINT y = 0; y < H; ++y) {
                const uint8_t* row = base + y * mapped.RowPitch;
                for (UINT x = 0; x < W; ++x) fwrite(row + x * 4, 1, 3, ppm);
            }
            fclose(ppm);
        }
        ID3D11DeviceContext_Unmap(ctx, (ID3D11Resource*)stg, 0);
    }
    fprintf(logf, "DX11_DEFERRED_MRT_COMPLETE!\n");
    fclose(logf);
    return 0;
}
