// dx12_smrast_test.c — Native DirectX 12 (ID3D12Device) 3D Lit Textured MRT Test for SmRast-VK v0.4
// Runs via Proton 11 vkd3d-proton (d3d12.dll + d3d12core.dll) -> SmRast-VK Compute Rasterizer -> H20 78 SMs!
// Compile on Linux: x86_64-w64-mingw32-gcc -O2 bench/dx12_smrast_test.c -o bench/dx12_smrast_test.exe -ld3d12 -ldxgi -ld3dcompiler_47 -lole32

#define COBJMACROS
#define WIDL_C_INLINE_WRAPPERS
#define INITGUID
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define W 512
#define H 512
#define CHECK_HR(hr, msg) do { HRESULT _h = (hr); if (FAILED(_h)) { fprintf(stderr, "[DX12] FAIL %s: hr=0x%08lx\n", msg, (unsigned long)_h); return 1; } } while (0)

typedef struct { float px, py, pz; float nx, ny, nz; float u, v; } Vertex;

static void mat4_identity(float m[16]) { memset(m, 0, 64); m[0]=m[5]=m[10]=m[15]=1.f; }
static void mat4_mul(float o[16], const float a[16], const float b[16]) {
    float t[16];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            t[r*4+c] = a[r*4+0]*b[0*4+c] + a[r*4+1]*b[1*4+c] + a[r*4+2]*b[2*4+c] + a[r*4+3]*b[3*4+c];
    memcpy(o, t, 64);
}
static void mat4_persp(float m[16], float fov_rad, float aspect, float zn, float zf) {
    memset(m, 0, 64);
    float f = 1.0f / tanf(fov_rad * 0.5f);
    m[0] = f / aspect; m[5] = f;
    m[10] = zf / (zf - zn); m[11] = 1.0f;
    m[14] = (-zn * zf) / (zf - zn);
}
static void mat4_roty(float m[16], float a) {
    mat4_identity(m);
    float c = cosf(a), s = sinf(a);
    m[0] = c; m[2] = -s; m[8] = s; m[10] = c;
}
static void mat4_rotx(float m[16], float a) {
    mat4_identity(m);
    float c = cosf(a), s = sinf(a);
    m[5] = c; m[6] = s; m[9] = -s; m[10] = c;
}
static void mat4_trans(float m[16], float x, float y, float z) {
    mat4_identity(m); m[12]=x; m[13]=y; m[14]=z;
}

static const char* kHLSL =
"cbuffer SceneCB : register(b0) {\n"
"    row_major float4x4 mvp;\n"
"    row_major float4x4 model;\n"
"    float4 lightDir;\n"
"    float4 tint;\n"
"};\n"
"Texture2D texAlbedo : register(t0);\n"
"SamplerState sampLin : register(s0);\n"
"struct VSIn { float3 pos : POSITION; float3 nrm : NORMAL; float2 uv : TEXCOORD; };\n"
"struct PSIn { float4 pos : SV_POSITION; float3 wn : TEXCOORD0; float2 uv : TEXCOORD1; };\n"
"struct PSOut { float4 color : SV_Target0; float4 gnorm : SV_Target1; };\n"
"PSIn VSMain(VSIn i) {\n"
"    PSIn o;\n"
"    o.pos = mul(float4(i.pos, 1.0), mvp);\n"
"    o.wn  = normalize(mul(float4(i.nrm, 0.0), model).xyz);\n"
"    o.uv  = i.uv;\n"
"    return o;\n"
"}\n"
"PSOut PSMain(PSIn i) {\n"
"    float3 N = normalize(i.wn);\n"
"    float3 L = normalize(lightDir.xyz);\n"
"    float3 V = float3(0.0, 0.0, -1.0);\n"
"    float3 H = normalize(L + V);\n"
"    float ndl = max(dot(N, L), 0.0);\n"
"    float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.6;\n"
"    float3 alb = texAlbedo.Sample(sampLin, i.uv).rgb * tint.rgb;\n"
"    float3 col = alb * (0.18 + 0.85 * ndl) + float3(1.0, 0.95, 0.85) * spec;\n"
"    PSOut o;\n"
"    o.color = float4(saturate(col), 1.0);\n"
"    o.gnorm = float4(N * 0.5 + 0.5, 1.0);\n"
"    return o;\n"
"}\n";

static void add_quad(Vertex* v, uint16_t* idx, int* vc, int* ic,
                     float p0[3], float p1[3], float p2[3], float p3[3], float n[3]) {
    uint16_t b = (uint16_t)(*vc);
    v[b+0] = (Vertex){p0[0],p0[1],p0[2], n[0],n[1],n[2], 0,0};
    v[b+1] = (Vertex){p1[0],p1[1],p1[2], n[0],n[1],n[2], 1,0};
    v[b+2] = (Vertex){p2[0],p2[1],p2[2], n[0],n[1],n[2], 1,1};
    v[b+3] = (Vertex){p3[0],p3[1],p3[2], n[0],n[1],n[2], 0,1};
    idx[(*ic)++] = b+0; idx[(*ic)++] = b+1; idx[(*ic)++] = b+2;
    idx[(*ic)++] = b+0; idx[(*ic)++] = b+2; idx[(*ic)++] = b+3;
    *vc += 4;
}

static void build_cube(Vertex* v, uint16_t* idx, int* vc, int* ic) {
    float s = 0.85f;
    float f0[3]={-s,-s,-s}, f1[3]={s,-s,-s}, f2[3]={s,s,-s}, f3[3]={-s,s,-s}; float nf[3]={0,0,-1};
    float b0[3]={s,-s,s},   b1[3]={-s,-s,s}, b2[3]={-s,s,s}, b3[3]={s,s,s};   float nb[3]={0,0,1};
    float l0[3]={-s,-s,s},  l1[3]={-s,-s,-s},l2[3]={-s,s,-s},l3[3]={-s,s,s};  float nl[3]={-1,0,0};
    float r0[3]={s,-s,-s},  r1[3]={s,-s,s},  r2[3]={s,s,s},  r3[3]={s,s,-s};  float nr[3]={1,0,0};
    float t0[3]={-s,s,-s},  t1[3]={s,s,-s},  t2[3]={s,s,s},  t3[3]={-s,s,s};  float nt[3]={0,1,0};
    float d0[3]={-s,-s,s},  d1[3]={s,-s,s},  d2[3]={s,-s,-s},d3[3]={-s,-s,-s};float nd[3]={0,-1,0};
    add_quad(v,idx,vc,ic, f0,f1,f2,f3, nf);
    add_quad(v,idx,vc,ic, b0,b1,b2,b3, nb);
    add_quad(v,idx,vc,ic, l0,l1,l2,l3, nl);
    add_quad(v,idx,vc,ic, r0,r1,r2,r3, nr);
    add_quad(v,idx,vc,ic, t0,t1,t2,t3, nt);
    add_quad(v,idx,vc,ic, d0,d1,d2,d3, nd);
}

static HRESULT create_committed_buf(ID3D12Device* dev, D3D12_HEAP_TYPE htype, uint64_t sz, D3D12_RESOURCE_STATES st, ID3D12Resource** out) {
    D3D12_HEAP_PROPERTIES hp = { htype, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1 };
    D3D12_RESOURCE_DESC rd;
    memset(&rd, 0, sizeof(rd));
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = sz; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
    rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return ID3D12Device_CreateCommittedResource(dev, &hp, D3D12_HEAP_FLAG_NONE, &rd, st, NULL, &IID_ID3D12Resource, (void**)out);
}

int main(void) {
    printf("[DX12] Starting SmRast-VK DirectX 12 (vkd3d-proton) 3D MRT Test...\n");
    fflush(stdout);

    IDXGIFactory4* fac = NULL;
    CHECK_HR(CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&fac), "CreateDXGIFactory1");
    IDXGIAdapter1* adapter = NULL;
    CHECK_HR(IDXGIFactory4_EnumAdapters1(fac, 0, &adapter), "EnumAdapters1");
    DXGI_ADAPTER_DESC1 ad;
    IDXGIAdapter1_GetDesc1(adapter, &ad);
    char adname[128];
    wcstombs(adname, ad.Description, sizeof(adname));
    printf("[DX12] Adapter: %s (VRAM=%llu MB)\n", adname, (unsigned long long)(ad.DedicatedVideoMemory >> 20));
    fflush(stdout);

    ID3D12Device* dev = NULL;
    CHECK_HR(D3D12CreateDevice((IUnknown*)adapter, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void**)&dev), "D3D12CreateDevice");
    printf("[DX12] D3D12CreateDevice OK!\n");
    fflush(stdout);

    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0 };
    ID3D12CommandQueue* cq = NULL;
    CHECK_HR(ID3D12Device_CreateCommandQueue(dev, &qd, &IID_ID3D12CommandQueue, (void**)&cq), "CreateCommandQueue");

    ID3D12CommandAllocator* ca = NULL;
    CHECK_HR(ID3D12Device_CreateCommandAllocator(dev, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator, (void**)&ca), "CreateCommandAllocator");
    ID3D12GraphicsCommandList* cl = NULL;
    CHECK_HR(ID3D12Device_CreateCommandList(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, ca, NULL, &IID_ID3D12GraphicsCommandList, (void**)&cl), "CreateCommandList");

    ID3D12Fence* fence = NULL;
    CHECK_HR(ID3D12Device_CreateFence(dev, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void**)&fence), "CreateFence");
    HANDLE fev = CreateEventA(NULL, FALSE, FALSE, NULL);

    // Compile HLSL VS + PS (SM 5.0 DXBC -> vkd3d-proton -> SPIR-V -> SmRast-VK Compute Shaders!)
    ID3DBlob *vsb = NULL, *psb = NULL, *errb = NULL;
    HRESULT hr = D3DCompile(kHLSL, strlen(kHLSL), "dx12_test.hlsl", NULL, NULL, "VSMain", "vs_5_0", 0, 0, &vsb, &errb);
    if (FAILED(hr)) { fprintf(stderr, "VS compile err: %s\n", errb ? (char*)ID3D10Blob_GetBufferPointer(errb) : "?"); return 1; }
    hr = D3DCompile(kHLSL, strlen(kHLSL), "dx12_test.hlsl", NULL, NULL, "PSMain", "ps_5_0", 0, 0, &psb, &errb);
    if (FAILED(hr)) { fprintf(stderr, "PS compile err: %s\n", errb ? (char*)ID3D10Blob_GetBufferPointer(errb) : "?"); return 1; }

    // Root Signature: param0 = Root CBV (b0), param1 = SRV Descriptor Table (t0), static sampler s0
    D3D12_DESCRIPTOR_RANGE srv_range = { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0 };
    D3D12_ROOT_PARAMETER rps[2];
    memset(rps, 0, sizeof(rps));
    rps[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rps[0].Descriptor.ShaderRegister = 0; rps[0].Descriptor.RegisterSpace = 0;
    rps[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rps[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rps[1].DescriptorTable.NumDescriptorRanges = 1;
    rps[1].DescriptorTable.pDescriptorRanges = &srv_range;
    rps[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC ss;
    memset(&ss, 0, sizeof(ss));
    ss.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    ss.AddressU = ss.AddressV = ss.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    ss.MaxLOD = 1000.f; ss.ShaderRegister = 0; ss.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rsd = { 2, rps, 1, &ss, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT };
    ID3DBlob* rsb = NULL;
    CHECK_HR(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &rsb, &errb), "SerializeRootSignature");
    ID3D12RootSignature* root_sig = NULL;
    CHECK_HR(ID3D12Device_CreateRootSignature(dev, 0, ID3D10Blob_GetBufferPointer(rsb), ID3D10Blob_GetBufferSize(rsb), &IID_ID3D12RootSignature, (void**)&root_sig), "CreateRootSignature");

    // Graphics PSO: 2x MRT (RGBA8 + RGBA16F) + D32F Depth
    D3D12_INPUT_ELEMENT_DESC ied[3] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,  D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psod;
    memset(&psod, 0, sizeof(psod));
    psod.pRootSignature = root_sig;
    psod.VS.pShaderBytecode = ID3D10Blob_GetBufferPointer(vsb); psod.VS.BytecodeLength = ID3D10Blob_GetBufferSize(vsb);
    psod.PS.pShaderBytecode = ID3D10Blob_GetBufferPointer(psb); psod.PS.BytecodeLength = ID3D10Blob_GetBufferSize(psb);
    psod.SampleMask = 0xFFFFFFFFu;
    psod.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psod.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psod.RasterizerState.DepthClipEnable = TRUE;
    psod.BlendState.RenderTarget[0].RenderTargetWriteMask = 0xF;
    psod.BlendState.RenderTarget[1].RenderTargetWriteMask = 0xF;
    psod.DepthStencilState.DepthEnable = TRUE;
    psod.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    psod.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    psod.InputLayout.pInputElementDescs = ied; psod.InputLayout.NumElements = 3;
    psod.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psod.NumRenderTargets = 2;
    psod.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psod.RTVFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
    psod.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    psod.SampleDesc.Count = 1;

    ID3D12PipelineState* pso = NULL;
    CHECK_HR(ID3D12Device_CreateGraphicsPipelineState(dev, &psod, &IID_ID3D12PipelineState, (void**)&pso), "CreateGraphicsPipelineState");
    printf("[DX12] CreateGraphicsPipelineState (2x MRT + D32F + SRV) OK!\n");
    fflush(stdout);

    // Descriptor heaps: RTV (2), DSV (1), CBV_SRV_UAV (1 shader visible)
    ID3D12DescriptorHeap *rtv_heap = NULL, *dsv_heap = NULL, *srv_heap = NULL;
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0 };
    CHECK_HR(ID3D12Device_CreateDescriptorHeap(dev, &hd, &IID_ID3D12DescriptorHeap, (void**)&rtv_heap), "CreateRTVHeap");
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV; hd.NumDescriptors = 1;
    CHECK_HR(ID3D12Device_CreateDescriptorHeap(dev, &hd, &IID_ID3D12DescriptorHeap, (void**)&dsv_heap), "CreateDSVHeap");
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CHECK_HR(ID3D12Device_CreateDescriptorHeap(dev, &hd, &IID_ID3D12DescriptorHeap, (void**)&srv_heap), "CreateSRVHeap");

    // Create RT0 (RGBA8), RT1 (RGBA16F), Depth (D32F), Texture (128x128 RGBA8 checker)
    D3D12_HEAP_PROPERTIES def_hp = { D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1 };
    ID3D12Resource *rt0 = NULL, *rt1 = NULL, *dsv_res = NULL, *tex = NULL;
    D3D12_RESOURCE_DESC td;
    memset(&td, 0, sizeof(td));
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = W; td.Height = H; td.DepthOrArraySize = 1; td.MipLevels = 1;
    td.SampleDesc.Count = 1; td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    D3D12_CLEAR_VALUE cv0; memset(&cv0, 0, sizeof(cv0));
    cv0.Format = DXGI_FORMAT_R8G8B8A8_UNORM; cv0.Color[0]=0.05f; cv0.Color[1]=0.08f; cv0.Color[2]=0.18f; cv0.Color[3]=1.0f;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    CHECK_HR(ID3D12Device_CreateCommittedResource(dev, &def_hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_RENDER_TARGET, &cv0, &IID_ID3D12Resource, (void**)&rt0), "CreateRT0");

    D3D12_CLEAR_VALUE cv1; memset(&cv1, 0, sizeof(cv1)); cv1.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    CHECK_HR(ID3D12Device_CreateCommittedResource(dev, &def_hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_RENDER_TARGET, &cv1, &IID_ID3D12Resource, (void**)&rt1), "CreateRT1");

    D3D12_CLEAR_VALUE cvd; memset(&cvd, 0, sizeof(cvd)); cvd.Format = DXGI_FORMAT_D32_FLOAT; cvd.DepthStencil.Depth = 1.0f;
    td.Format = DXGI_FORMAT_D32_FLOAT; td.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    CHECK_HR(ID3D12Device_CreateCommittedResource(dev, &def_hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_DEPTH_WRITE, &cvd, &IID_ID3D12Resource, (void**)&dsv_res), "CreateDSV");

    td.Width = 128; td.Height = 128; td.Format = DXGI_FORMAT_R8G8B8A8_UNORM; td.Flags = D3D12_RESOURCE_FLAG_NONE;
    CHECK_HR(ID3D12Device_CreateCommittedResource(dev, &def_hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COPY_DEST, NULL, &IID_ID3D12Resource, (void**)&tex), "CreateTex");

    UINT rtv_inc = ID3D12Device_GetDescriptorHandleIncrementSize(dev, D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE rtv0_h = ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(rtv_heap);
    D3D12_CPU_DESCRIPTOR_HANDLE rtv1_h;
    rtv1_h.ptr = rtv0_h.ptr + rtv_inc;
    ID3D12Device_CreateRenderTargetView(dev, rt0, NULL, rtv0_h);
    ID3D12Device_CreateRenderTargetView(dev, rt1, NULL, rtv1_h);
    D3D12_CPU_DESCRIPTOR_HANDLE dsv_h = ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(dsv_heap);
    ID3D12Device_CreateDepthStencilView(dev, dsv_res, NULL, dsv_h);
    D3D12_CPU_DESCRIPTOR_HANDLE srv_cpu_h = ID3D12DescriptorHeap_GetCPUDescriptorHandleForHeapStart(srv_heap);
    D3D12_SHADER_RESOURCE_VIEW_DESC srvd; memset(&srvd, 0, sizeof(srvd));
    srvd.Format = DXGI_FORMAT_R8G8B8A8_UNORM; srvd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; srvd.Texture2D.MipLevels = 1;
    ID3D12Device_CreateShaderResourceView(dev, tex, &srvd, srv_cpu_h);

    // Upload buffers: VB, IB, CB, TexUpload, Readback
    Vertex verts[24]; uint16_t idxs[36]; int vc = 0, ic = 0;
    build_cube(verts, idxs, &vc, &ic);
    ID3D12Resource *vb = NULL, *ib = NULL, *cb = NULL, *tex_up = NULL, *rb = NULL;
    CHECK_HR(create_committed_buf(dev, D3D12_HEAP_TYPE_UPLOAD, sizeof(verts), D3D12_RESOURCE_STATE_GENERIC_READ, &vb), "VB");
    CHECK_HR(create_committed_buf(dev, D3D12_HEAP_TYPE_UPLOAD, sizeof(idxs), D3D12_RESOURCE_STATE_GENERIC_READ, &ib), "IB");
    CHECK_HR(create_committed_buf(dev, D3D12_HEAP_TYPE_UPLOAD, 512, D3D12_RESOURCE_STATE_GENERIC_READ, &cb), "CB");
    CHECK_HR(create_committed_buf(dev, D3D12_HEAP_TYPE_UPLOAD, 128 * 512, D3D12_RESOURCE_STATE_GENERIC_READ, &tex_up), "TexUp");
    CHECK_HR(create_committed_buf(dev, D3D12_HEAP_TYPE_READBACK, (uint64_t)W * H * 4, D3D12_RESOURCE_STATE_COPY_DEST, &rb), "RB");

    void* p = NULL;
    ID3D12Resource_Map(vb, 0, NULL, &p); memcpy(p, verts, sizeof(verts)); ID3D12Resource_Unmap(vb, 0, NULL);
    ID3D12Resource_Map(ib, 0, NULL, &p); memcpy(p, idxs, sizeof(idxs));   ID3D12Resource_Unmap(ib, 0, NULL);

    // Procedural gold/cobalt checkerboard texture
    ID3D12Resource_Map(tex_up, 0, NULL, &p);
    uint8_t* tp = (uint8_t*)p;
    for (int y = 0; y < 128; ++y) {
        for (int x = 0; x < 128; ++x) {
            int chk = ((x >> 4) ^ (y >> 4)) & 1;
            uint8_t* px = tp + y * 512 + x * 4;
            px[0] = chk ? 245 : 40;
            px[1] = chk ? 190 : 140;
            px[2] = chk ? 65  : 235;
            px[3] = 255;
        }
    }
    ID3D12Resource_Unmap(tex_up, 0, NULL);

    // Fill SceneCB (MVP + Model + Light + Tint)
    float ry[16], rx[16], tr[16], mdl[16], view_m[16], proj[16], mv[16], mvp[16];
    mat4_roty(ry, 0.62f); mat4_rotx(rx, 0.42f); mat4_mul(mdl, rx, ry);
    mat4_trans(view_m, 0.0f, 0.0f, 3.1f);
    mat4_persp(proj, 0.95f, 1.0f, 0.1f, 50.0f);
    mat4_mul(mv, mdl, view_m);
    mat4_mul(mvp, mv, proj);
    struct { float mvp[16]; float model[16]; float light[4]; float tint[4]; } cb_data;
    memcpy(cb_data.mvp, mvp, 64);
    memcpy(cb_data.model, mdl, 64);
    cb_data.light[0]=0.5f; cb_data.light[1]=0.7f; cb_data.light[2]=-0.6f; cb_data.light[3]=0.0f;
    cb_data.tint[0]=1.0f;  cb_data.tint[1]=1.0f;  cb_data.tint[2]=1.0f;   cb_data.tint[3]=1.0f;
    ID3D12Resource_Map(cb, 0, NULL, &p); memcpy(p, &cb_data, sizeof(cb_data)); ID3D12Resource_Unmap(cb, 0, NULL);

    // Copy texture upload -> default texture
    D3D12_TEXTURE_COPY_LOCATION dst_loc = { tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, { 0 } };
    D3D12_TEXTURE_COPY_LOCATION src_loc;
    memset(&src_loc, 0, sizeof(src_loc));
    src_loc.pResource = tex_up; src_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src_loc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    src_loc.PlacedFootprint.Footprint.Width = 128; src_loc.PlacedFootprint.Footprint.Height = 128;
    src_loc.PlacedFootprint.Footprint.Depth = 1; src_loc.PlacedFootprint.Footprint.RowPitch = 512;
    ID3D12GraphicsCommandList_CopyTextureRegion(cl, &dst_loc, 0, 0, 0, &src_loc, NULL);

    D3D12_RESOURCE_BARRIER bar;
    memset(&bar, 0, sizeof(bar));
    bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Transition.pResource = tex;
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    bar.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ID3D12GraphicsCommandList_ResourceBarrier(cl, 1, &bar);

    // Render 2x MRT + Depth
    D3D12_CPU_DESCRIPTOR_HANDLE rtvs[2] = { rtv0_h, rtv1_h };
    ID3D12GraphicsCommandList_OMSetRenderTargets(cl, 2, rtvs, FALSE, &dsv_h);
    ID3D12GraphicsCommandList_ClearRenderTargetView(cl, rtv0_h, cv0.Color, 0, NULL);
    float z4[4] = {0,0,0,0};
    ID3D12GraphicsCommandList_ClearRenderTargetView(cl, rtv1_h, z4, 0, NULL);
    ID3D12GraphicsCommandList_ClearDepthStencilView(cl, dsv_h, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, NULL);

    D3D12_VIEWPORT vp = { 0.f, 0.f, (float)W, (float)H, 0.f, 1.f };
    D3D12_RECT sc = { 0, 0, W, H };
    ID3D12GraphicsCommandList_RSSetViewports(cl, 1, &vp);
    ID3D12GraphicsCommandList_RSSetScissorRects(cl, 1, &sc);
    ID3D12GraphicsCommandList_SetPipelineState(cl, pso);
    ID3D12GraphicsCommandList_SetGraphicsRootSignature(cl, root_sig);
    ID3D12GraphicsCommandList_SetDescriptorHeaps(cl, 1, &srv_heap);
    ID3D12GraphicsCommandList_SetGraphicsRootConstantBufferView(cl, 0, ID3D12Resource_GetGPUVirtualAddress(cb));
    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu_h = ID3D12DescriptorHeap_GetGPUDescriptorHandleForHeapStart(srv_heap);
    ID3D12GraphicsCommandList_SetGraphicsRootDescriptorTable(cl, 1, srv_gpu_h);

    D3D12_VERTEX_BUFFER_VIEW vbv = { ID3D12Resource_GetGPUVirtualAddress(vb), sizeof(verts), sizeof(Vertex) };
    D3D12_INDEX_BUFFER_VIEW  ibv = { ID3D12Resource_GetGPUVirtualAddress(ib), sizeof(idxs), DXGI_FORMAT_R16_UINT };
    ID3D12GraphicsCommandList_IASetPrimitiveTopology(cl, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D12GraphicsCommandList_IASetVertexBuffers(cl, 0, 1, &vbv);
    ID3D12GraphicsCommandList_IASetIndexBuffer(cl, &ibv);
    ID3D12GraphicsCommandList_DrawIndexedInstanced(cl, 36, 1, 0, 0, 0);

    // Transition RT0 -> COPY_SOURCE and copy to readback buffer
    bar.Transition.pResource = rt0;
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    ID3D12GraphicsCommandList_ResourceBarrier(cl, 1, &bar);

    D3D12_TEXTURE_COPY_LOCATION rb_dst;
    memset(&rb_dst, 0, sizeof(rb_dst));
    rb_dst.pResource = rb; rb_dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    rb_dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rb_dst.PlacedFootprint.Footprint.Width = W; rb_dst.PlacedFootprint.Footprint.Height = H;
    rb_dst.PlacedFootprint.Footprint.Depth = 1; rb_dst.PlacedFootprint.Footprint.RowPitch = W * 4;
    D3D12_TEXTURE_COPY_LOCATION rt_src = { rt0, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, { 0 } };
    ID3D12GraphicsCommandList_CopyTextureRegion(cl, &rb_dst, 0, 0, 0, &rt_src, NULL);

    CHECK_HR(ID3D12GraphicsCommandList_Close(cl), "Close");
    ID3D12CommandList* lists[1] = { (ID3D12CommandList*)cl };
    ID3D12CommandQueue_ExecuteCommandLists(cq, 1, lists);
    CHECK_HR(ID3D12CommandQueue_Signal(cq, fence, 1), "Signal");
    if (ID3D12Fence_GetCompletedValue(fence) < 1) {
        CHECK_HR(ID3D12Fence_SetEventOnCompletion(fence, 1, fev), "SetEventOnCompletion");
        WaitForSingleObject(fev, 15000);
    }

    // Verify rendered pixels & dump PPM
    ID3D12Resource_Map(rb, 0, NULL, &p);
    const uint8_t* img = (const uint8_t*)p;
    int bg_idx = (4 * W + 4) * 4;
    int mid_idx = ((H/2) * W + (W/2)) * 4;
    int lit_px = 0;
    for (int i = 0; i < W * H; ++i) {
        if (img[i*4+0] > 40 || img[i*4+1] > 40 || img[i*4+2] > 70) lit_px++;
    }
    FILE* f = fopen("Z:\\tmp\\dx12_smrast_h20.ppm", "wb");
    if (f) {
        fprintf(f, "P6\n%d %d\n255\n", W, H);
        for (int i = 0; i < W * H; ++i) {
            uint8_t rgb[3] = { img[i*4+0], img[i*4+1], img[i*4+2] };
            fwrite(rgb, 1, 3, f);
        }
        fclose(f);
    }
    FILE* rf = fopen("Z:\\tmp\\dx12_smrast_result.txt", "w");
    int pass = (lit_px > 20000 && (img[mid_idx+0] > 30 || img[mid_idx+1] > 30 || img[mid_idx+2] > 60));
    char summary[512];
    snprintf(summary, sizeof(summary),
             "DX12_VKD3D_TEST %s | adapter=\"%s\" | lit_px=%d/%d | BG=(%u,%u,%u) CENTER=(%u,%u,%u)\n",
             pass ? "PASS" : "FAIL", adname, lit_px, W*H,
             img[bg_idx+0], img[bg_idx+1], img[bg_idx+2],
             img[mid_idx+0], img[mid_idx+1], img[mid_idx+2]);
    printf("%s", summary);
    if (rf) { fputs(summary, rf); fclose(rf); }
    ID3D12Resource_Unmap(rb, 0, NULL);
    return pass ? 0 : 1;
}
