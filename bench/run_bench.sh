#!/usr/bin/env bash
set -euo pipefail
cd .
rm -f /tmp/smrast_cache_*
bash build.sh

echo "=== 1. VULKAN 1.3 GAME ENGINE FEATURE TEST (DynamicRendering + ExtDynState + Sync2 + 256 Instancing + RTT->Strip) ==="
VK_ICD_FILENAMES=$PWD/smrast_icd.json \
SMRAST_SHADER_DIR=./build \
./build/test_vk13_game_features 2>&1

echo "=== 2. H20 SmRast-VK v0.2 (78 SMs, pure Compute qf[2]) 1080p PBR BENCHMARK ==="
for T in 100000 500000 1000000 2000000; do
    OUTARG=""
    if [ "$T" -eq 1000000 ]; then OUTARG="--out /tmp/pbr_1m_h20.ppm"; fi
    VK_ICD_FILENAMES=$PWD/smrast_icd.json \
    SMRAST_SHADER_DIR=./build \
    ./build/smrast_stress_bench --tris "$T" --warmup 5 --frames 40 $OUTARG 2>&1 | grep -E "RESULT|ERROR|ERR"
done

echo "=== 3. CPU LAVAPIPE (Mesa Software Rasterizer) 1080p PBR BENCHMARK ==="
for T in 100000 500000 1000000 2000000; do
    OUTARG=""
    FR=10
    if [ "$T" -eq 1000000 ]; then OUTARG="--out /tmp/pbr_1m_lvp.ppm"; FR=6; fi
    if [ "$T" -eq 2000000 ]; then FR=4; fi
    VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
    ./build/smrast_stress_bench --tris "$T" --warmup 2 --frames "$FR" $OUTARG 2>&1 | grep -E "RESULT|ERROR|ERR"
done

ffmpeg -y -i /tmp/pbr_1m_h20.ppm /tmp/pbr_1m_h20.png >/dev/null 2>&1
ffmpeg -y -i /tmp/pbr_1m_lvp.ppm /tmp/pbr_1m_lvp.png >/dev/null 2>&1
python3 - << 'PYEOF'
import numpy as np
from PIL import Image
a = np.asarray(Image.open('/tmp/pbr_1m_h20.png'), dtype=np.float32)
b = np.asarray(Image.open('/tmp/pbr_1m_lvp.png'), dtype=np.float32)
mae = float(np.mean(np.abs(a - b)))
mse = float(np.mean((a - b) ** 2))
psnr = 20.0 * np.log10(255.0 / np.sqrt(max(mse, 1e-9)))
print(f"PIXEL PARITY H20_SmRast vs Lavapipe (1920x1080, 1M tris): MAE={mae:.2f}/255, PSNR={psnr:.2f} dB")
PYEOF
base64 -w0 /tmp/pbr_1m_h20.png > /tmp/pbr_1m_h20.png.b64
base64 -w0 /tmp/pbr_1m_lvp.png > /tmp/pbr_1m_lvp.png.b64
ls -lh /tmp/pbr_1m_h20.png /tmp/pbr_1m_lvp.png
