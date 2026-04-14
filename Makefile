.PHONY: help build run benchmark plot compare clean

help:
	@echo "Targets:"
	@echo "  make build      - Configure and build C/C++ targets via CMake"
	@echo "  make run        - Run all implementations (placeholder script)"
	@echo "  make benchmark  - Run benchmark sweep (placeholder script)"
	@echo "  make plot       - Generate plots using uv-managed Python"
	@echo "  make compare    - Compare output images across implementations"
	@echo "  make clean      - Remove CMake build artifacts"

build:
	@./scripts/build_all

run:
	@./scripts/run_all

benchmark:
	@./scripts/benchmark_all

plot:
	@./scripts/plot_all

compare:
	@./scripts/compare_outputs

clean:
	@rm -rf build
