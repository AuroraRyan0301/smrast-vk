CXX := g++
CXXFLAGS := -O3 -std=c++17 -fPIC -Wall -Wextra -Wno-missing-field-initializers -Wno-unused-parameter -Wno-unused-function -Iinclude -I/usr/include/spirv_cross
LDFLAGS := -shared -Wl,--no-undefined -ldl -lspirv-cross-c-shared -lshaderc -lxcb -lxcb-shm -lX11 -lXext -lpthread
GLSLANG := glslangValidator

BUILD_DIR := build
TARGET := $(BUILD_DIR)/libvulkan_smrast.so
SHADERS := $(BUILD_DIR)/smrast_clear.comp.spv $(BUILD_DIR)/smrast_bin.comp.spv $(BUILD_DIR)/smrast_prim_bin.comp.spv $(BUILD_DIR)/smrast_fine.comp.spv $(BUILD_DIR)/smrast_indirect_prep.comp.spv $(BUILD_DIR)/smrast_depth_only.comp.spv $(BUILD_DIR)/smrast_blit.comp.spv $(BUILD_DIR)/smrast_hiz_update.comp.spv
SRCS := src/smrast_icd.cpp src/smrast_raster_engine.cpp src/spirv_lower.cpp src/smrast_wsi.cpp
OBJS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

all: $(TARGET) $(SHADERS) bench

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.cpp include/smrast_internal.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.comp.spv: shaders/%.comp | $(BUILD_DIR)
	$(GLSLANG) --target-env vulkan1.3 -V $< -o $@

bench: $(BUILD_DIR)/smrast_stress_bench $(BUILD_DIR)/test_vk13_game_features $(BUILD_DIR)/test_v04_features $(BUILD_DIR)/test_v05_features $(BUILD_DIR)/test_v06_features $(BUILD_DIR)/test_v07_features

$(BUILD_DIR)/test_v04_features: bench/test_v04_features.cpp | $(BUILD_DIR)
	$(CXX) -O3 -std=c++17 bench/test_v04_features.cpp -o $@ -lvulkan -lm

$(BUILD_DIR)/test_v05_features: bench/test_v05_features.cpp | $(BUILD_DIR)
	$(CXX) -O3 -std=c++17 bench/test_v05_features.cpp -o $@ -lvulkan -lm

$(BUILD_DIR)/test_v06_features: bench/test_v06_features.cpp | $(BUILD_DIR)
	$(CXX) -O3 -std=c++17 bench/test_v06_features.cpp -o $@ -lvulkan -lm

$(BUILD_DIR)/test_v07_features: bench/test_v07_features.cpp | $(BUILD_DIR)
	$(CXX) -O3 -std=c++17 bench/test_v07_features.cpp -o $@ -lvulkan -lm

$(BUILD_DIR)/smrast_stress_bench: bench/smrast_stress_bench.cpp bench/pbr.vert bench/pbr.frag | $(BUILD_DIR)
	$(GLSLANG) -V bench/pbr.vert -o bench/pbr.vert.spv
	$(GLSLANG) -V bench/pbr.frag -o bench/pbr.frag.spv
	$(CXX) -O3 -std=c++17 bench/smrast_stress_bench.cpp -o $@ -lvulkan -lm

$(BUILD_DIR)/test_vk13_game_features: bench/test_vk13_game_features.cpp bench/inst.vert bench/inst.frag bench/post.vert bench/post.frag | $(BUILD_DIR)
	$(GLSLANG) -V bench/inst.vert -o bench/inst.vert.spv
	$(GLSLANG) -V bench/inst.frag -o bench/inst.frag.spv
	$(GLSLANG) -V bench/post.vert -o bench/post.vert.spv
	$(GLSLANG) -V bench/post.frag -o bench/post.frag.spv
	$(CXX) -O3 -std=c++17 bench/test_vk13_game_features.cpp -o $@ -lvulkan -lm

clean:
	rm -rf $(BUILD_DIR) bench/*.spv
