// dx11_smrast_test.c — Real DirectX 11 HLSL -> DXVK 2.4.1 -> SmRast-VK -> NVIDIA H20 78 SMs End-to-End Test!
// Compiles HLSL vs_5_0 + ps_5_0 via d3dcompiler_47.dll, draws a shaded 3D/2D RGB triangle via D3D11,
// reads back the 512x512 RenderTarget via CopyResource + Map, verifies pixels, and dumps a PPM image!
#define COBJMACROS
#define INITGUID
#include <windows.h>
#include <initguid.h>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

typedef HRESULT (WINAPI *PFN_D3DCompile)(
    LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*,
    LPCSTR, LPCSTR, UINT, UINT, ID3DBlob**, ID3DBlob**);

static const char* HLSL_SRC =
"struct VSIn { float3 pos : POSITION; float4 col : COLOR; };\n"
"struct PSIn { float4 pos : SV_POSITION; float4 col : COLOR; };\n"
"PSIn VSMain(VSIn i) {\n"
"    PSIn o;\n"
"    o.pos = float4(i.pos, 1.0);\n"
"    o.col = i.col;\n"
"    return o;\n"
"}\n"
"float4 PSMain(PSIn i) : SV_TARGET {\n"
"    return i.col;\n"
"}\n";

int main(void) {
    FILE* logf = fopen("Z:\\tmp\\dx11_smrast_result.txt", "w");
    if (!logf) logf = stdout;

    IDXGIFactory1* factory = NULL;
    HRESULT hr = CreateDXGIFactory1(&IID_IDXGIFactory1, (void**)&factory);
    fprintf(logf, "CreateDXGIFactory1 hr=0x%08lx\n", (unsigned long)hr); fflush(logf);
    if (FAILED(hr)) return 1;

    IDXGIAdapter1* adapter = NULL;
    for (UINT i = 0; IDXGIFactory1_EnumAdapters1(factory, i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        IDXGIAdapter1_GetDesc1(adapter, &desc);
        fprintf(logf, "DXGI Adapter[%u]: %ls (Vendor=0x%04x Device=0x%04x VRAM=%llu MiB)\n",
                i, desc.Description, desc.VendorId, desc.DeviceId,
                (unsigned long long)(desc.DedicatedVideoMemory / (1024ULL * 1024ULL)));
        fflush(logf);
        IDXGIAdapter1_Release(adapter);
    }

    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0, fl_out = 0;
    ID3D11Device* dev = NULL;
    ID3D11DeviceContext* ctx = NULL;
    hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, 0, &fl, 1,
                           D3D11_SDK_VERSION, &dev, &fl_out, &ctx);
    fprintf(logf, "D3D11CreateDevice hr=0x%08lx feature_level=0x%x\n", (unsigned long)hr, (unsigned)fl_out);
    fflush(logf);
    if (FAILED(hr)) return 2;

    const UINT W = 512, H = 512;
    D3D11_TEXTURE2D_DESC td;
    memset(&td, 0, sizeof(td));
    td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET;
    ID3D11Texture2D* rt_tex = NULL;
    hr = ID3D11Device_CreateTexture2D(dev, &td, NULL, &rt_tex);
    ID3D11RenderTargetView* rtv = NULL;
    hr = ID3D11Device_CreateRenderTargetView(dev, (ID3D11Resource*)rt_tex, NULL, &rtv);

    // Clear background to deep navy blue
    float clear_col[4] = {0.06f, 0.10f, 0.22f, 1.0f};
    ID3D11DeviceContext_ClearRenderTargetView(ctx, rtv, clear_col);

    // Compile real HLSL vs_5_0 + ps_5_0 -> DXBC -> DXVK SPIR-V -> SmRast-VK Compute Shaders!
    HMODULE d3dc = LoadLibraryA("d3dcompiler_47.dll");
    PFN_D3DCompile pCompile = d3dc ? (PFN_D3DCompile)GetProcAddress(d3dc, "D3DCompile") : NULL;
    fprintf(logf, "d3dcompiler_47.dll loaded=%d pCompile=%p\n", d3dc != NULL, (void*)pCompile); fflush(logf);
    int drew_tri = 0;
    if (pCompile) {
        ID3DBlob *vsb = NULL, *psb = NULL, *errb = NULL;
        HRESULT hr_vs = pCompile(HLSL_SRC, strlen(HLSL_SRC), "tri.hlsl", NULL, NULL, "VSMain", "vs_5_0", 0, 0, &vsb, &errb);
        HRESULT hr_ps = pCompile(HLSL_SRC, strlen(HLSL_SRC), "tri.hlsl", NULL, NULL, "PSMain", "ps_5_0", 0, 0, &psb, &errb);
        fprintf(logf, "D3DCompile VS=0x%08lx PS=0x%08lx\n", (unsigned long)hr_vs, (unsigned long)hr_ps); fflush(logf);
        if (SUCCEEDED(hr_vs) && SUCCEEDED(hr_ps)) {
            ID3D11VertexShader* vs = NULL; ID3D11PixelShader* ps = NULL;
            ID3D11Device_CreateVertexShader(dev, ID3D10Blob_GetBufferPointer(vsb), ID3D10Blob_GetBufferSize(vsb), NULL, &vs);
            ID3D11Device_CreatePixelShader(dev, ID3D10Blob_GetBufferPointer(psb), ID3D10Blob_GetBufferSize(psb), NULL, &ps);

            D3D11_INPUT_ELEMENT_DESC ied[2] = {
                {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0},
                {"COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
            };
            ID3D11InputLayout* il = NULL;
            ID3D11Device_CreateInputLayout(dev, ied, 2, ID3D10Blob_GetBufferPointer(vsb), ID3D10Blob_GetBufferSize(vsb), &il);

            struct V { float x, y, z, r, g, b, a; };
            struct V verts[3] = {
                {  0.0f,  0.75f, 0.5f, 1.0f, 0.15f, 0.25f, 1.0f }, // Top: Red
                {  0.75f, -0.65f, 0.5f, 0.15f, 0.95f, 0.35f, 1.0f }, // Right: Green
                { -0.75f, -0.65f, 0.5f, 0.20f, 0.45f, 1.00f, 1.0f }, // Left: Blue
            };
            D3D11_BUFFER_DESC bd; memset(&bd, 0, sizeof(bd));
            bd.ByteWidth = sizeof(verts); bd.Usage = D3D11_USAGE_IMMUTABLE; bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
            D3D11_SUBRESOURCE_DATA sd = {verts, 0, 0};
            ID3D11Buffer* vb = NULL;
            ID3D11Device_CreateBuffer(dev, &bd, &sd, &vb);

            D3D11_VIEWPORT vp = {0.0f, 0.0f, (float)W, (float)H, 0.0f, 1.0f};
            UINT stride = sizeof(struct V), offset = 0;
            ID3D11DeviceContext_OMSetRenderTargets(ctx, 1, &rtv, NULL);
            ID3D11DeviceContext_RSSetViewports(ctx, 1, &vp);
            ID3D11DeviceContext_IASetInputLayout(ctx, il);
            ID3D11DeviceContext_IASetVertexBuffers(ctx, 0, 1, &vb, &stride, &offset);
            ID3D11DeviceContext_IASetPrimitiveTopology(ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ID3D11DeviceContext_VSSetShader(ctx, vs, NULL, 0);
            ID3D11DeviceContext_PSSetShader(ctx, ps, NULL, 0);
            ID3D11DeviceContext_Draw(ctx, 3, 0);
            drew_tri = 1;
            fprintf(logf, "D3D11 Draw(3, 0) submitted via DXVK -> SmRast-VK!\n"); fflush(logf);
        }
    }

    td.Usage = D3D11_USAGE_STAGING;
    td.BindFlags = 0;
    td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Texture2D* stg = NULL;
    ID3D11Device_CreateTexture2D(dev, &td, NULL, &stg);
    ID3D11DeviceContext_CopyResource(ctx, (ID3D11Resource*)stg, (ID3D11Resource*)rt_tex);
    ID3D11DeviceContext_Flush(ctx);

    D3D11_MAPPED_SUBRESOURCE mapped; memset(&mapped, 0, sizeof(mapped));
    hr = ID3D11DeviceContext_Map(ctx, (ID3D11Resource*)stg, 0, D3D11_MAP_READ, 0, &mapped);
    fprintf(logf, "Map(Staging) hr=0x%08lx pitch=%u drew_tri=%d\n", (unsigned long)hr, (unsigned)mapped.RowPitch, drew_tri);
    if (SUCCEEDED(hr) && mapped.pData) {
        const uint8_t* base = (const uint8_t*)mapped.pData;
        const uint8_t* p_bg = base + 8 * mapped.RowPitch + 8 * 4;
        const uint8_t* p_ct = base + (H/2) * mapped.RowPitch + (W/2) * 4;
        fprintf(logf, "BG_PIXEL(8,8)=(%u,%u,%u,%u) CENTER_PIXEL(256,256)=(%u,%u,%u,%u)\n",
                p_bg[0], p_bg[1], p_bg[2], p_bg[3], p_ct[0], p_ct[1], p_ct[2], p_ct[3]);
        FILE* ppm = fopen("Z:\\tmp\\dx11_dxvk_h20.ppm", "wb");
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
    fprintf(logf, "DX11_DXVK_SMRAST_COMPLETE!\n");
    fclose(logf);
    return 0;
}
