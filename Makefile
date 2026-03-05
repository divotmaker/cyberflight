SHADER_DIR := crates/cf-render/shaders

mode ?= raster

run:
	CF_RENDER_MODE=$(mode) cargo run -p cyberflight --bin cyberflight

build: shaders
	cargo build

test:
	cargo test --lib

test-gpu:
	cargo test --features gpu-tests

clippy:
	cargo clippy --workspace -- -D warnings

screenshots:
	cargo run -p cyberflight --example screenshot

clean:
	cargo clean

RT_SHADERS := $(SHADER_DIR)/rt.rgen.spv $(SHADER_DIR)/rt.rmiss.spv $(SHADER_DIR)/rt.rchit.spv
COMPOSITE_SHADERS := $(SHADER_DIR)/composite.vert.spv $(SHADER_DIR)/composite.frag.spv
shaders: $(SHADER_DIR)/grid.vert.spv $(SHADER_DIR)/grid.frag.spv $(RT_SHADERS) $(COMPOSITE_SHADERS)

$(SHADER_DIR)/%.vert.spv: $(SHADER_DIR)/%.vert
	glslc $< -o $@

$(SHADER_DIR)/%.frag.spv: $(SHADER_DIR)/%.frag
	glslc $< -o $@

$(SHADER_DIR)/%.rgen.spv: $(SHADER_DIR)/%.rgen
	glslc --target-env=vulkan1.3 $< -o $@

$(SHADER_DIR)/%.rmiss.spv: $(SHADER_DIR)/%.rmiss
	glslc --target-env=vulkan1.3 $< -o $@

$(SHADER_DIR)/%.rchit.spv: $(SHADER_DIR)/%.rchit
	glslc --target-env=vulkan1.3 $< -o $@

buildtools:
	@echo "Checking build tool dependencies..."
	@which glslc > /dev/null 2>&1 || (echo "glslc not found — install vulkan-tools or shaderc" && exit 1)
	@which cargo > /dev/null 2>&1 || (echo "cargo not found — install Rust toolchain" && exit 1)
	@echo "All build tools present."

.PHONY: run build test test-gpu clippy clean shaders screenshots buildtools
