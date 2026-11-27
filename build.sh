#!/usr/bin/env bash
set -euo pipefail
mkdir -p build
for s in smrast_clear smrast_bin smrast_fine; do
    glslangValidator -V --target-env vulkan1.3 "shaders/${s}.comp" -o "build/${s}.comp.spv"
done
g++ -O3 -std=c++17 -fPIC -shared -Wl,--no-undefined \
    -Iinclude -I/usr/include/spirv_cross \
    -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter -Wno-unused-function \
    -o build/libvulkan_smrast.so \
    src/smrast_icd.cpp src/smrast_raster_engine.cpp src/smrast_wsi.cpp src/spirv_lower.cpp \
    -lspirv-cross-c-shared -lxcb -lX11 -ldl -lpthread
if [ -f bench/smrast_stress_bench.cpp ]; then
    glslangValidator -V bench/pbr.vert -o bench/pbr.vert.spv
    glslangValidator -V bench/pbr.frag -o bench/pbr.frag.spv
    g++ -O3 -std=c++17 bench/smrast_stress_bench.cpp -o build/smrast_stress_bench -lvulkan -lm
fi
if [ -f bench/test_vk13_game_features.cpp ]; then
    for s in inst.vert inst.frag post.vert post.frag; do
        glslangValidator -V "bench/${s}" -o "bench/${s}.spv"
    done
    g++ -O3 -std=c++17 bench/test_vk13_game_features.cpp -o build/test_vk13_game_features -lvulkan -lm
fi
echo "Built build/libvulkan_smrast.so + shaders + benchmark + vk13 game test successfully!"
