# Multilayer Perceptron WebAssembly Build Configuration

.PHONY: all build build-wasm build-demo clean test install dev serve help

# Default target
all: build

# Install dependencies
install:
	@echo "Installing Node.js dependencies..."
	npm install
	@echo "Installing Rust dependencies..."
	cargo check

# Build WebAssembly module for web target
build-wasm:
	@echo "Building WebAssembly module..."
	wasm-pack build --target web --out-dir pkg --release

# Build WebAssembly module for Node.js target  
build-node:
	@echo "Building WebAssembly module for Node.js..."
	wasm-pack build --target nodejs --out-dir pkg-node --release

# Build WebAssembly module for bundlers
build-bundler:
	@echo "Building WebAssembly module for bundlers..."
	wasm-pack build --target bundler --out-dir pkg-bundler --release

# Build all targets
build: build-wasm build-node build-bundler

# Build demo website
build-demo: build-wasm
	@echo "Building demo website..."
	npm run build:demo

# Development server with hot reload
dev: build-wasm
	@echo "Starting development server..."
	npm run dev

# Simple HTTP server
serve:
	@echo "Starting HTTP server on port 8000..."
	@echo "Open http://localhost:8000/dist in your browser"
	npm run serve

# Run WebAssembly tests
test:
	@echo "Running WebAssembly tests..."
	wasm-pack test --headless --firefox

# Test with Chrome
test-chrome:
	@echo "Running WebAssembly tests in Chrome..."
	wasm-pack test --headless --chrome

# Run Rust tests
test-rust:
	@echo "Running Rust tests..."
	cargo test

# Run all tests
test-all: test-rust test

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf pkg pkg-node pkg-bundler dist node_modules
	cargo clean

# Lint Rust code
lint:
	@echo "Linting Rust code..."
	cargo clippy -- -D warnings
	cargo fmt --check

# Format Rust code
format:
	@echo "Formatting Rust code..."
	cargo fmt

# Check for security vulnerabilities
audit:
	@echo "Auditing dependencies..."
	cargo audit
	npm audit

# Build optimized release
release: clean
	@echo "Building optimized release..."
	CARGO_PROFILE_RELEASE_LTO=true \
	CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
	CARGO_PROFILE_RELEASE_PANIC=abort \
	wasm-pack build --target web --out-dir pkg --release
	wasm-pack build --target nodejs --out-dir pkg-node --release
	wasm-pack build --target bundler --out-dir pkg-bundler --release
	npm run build:demo

# Build for GitHub Pages deployment
build-pages: clean install
	@echo "Building for GitHub Pages deployment..."
	@echo "Building optimized WebAssembly..."
	wasm-pack build --target web --out-dir pkg --release
	@echo "Optimizing WebAssembly file size..."
	@if command -v wasm-opt >/dev/null 2>&1; then \
		wasm-opt -Os pkg/multilayer_perceptron_bg.wasm -o pkg/multilayer_perceptron_bg.wasm; \
		echo "WebAssembly optimized with wasm-opt"; \
	else \
		echo "wasm-opt not found, skipping optimization"; \
	fi
	@echo "Building production website..."
	npm run build:pages
	@echo "Build complete! Files are in dist/ directory"

# Deploy to GitHub Pages (local testing)
deploy-local: build-pages
	@echo "Starting local server for GitHub Pages testing..."
	@echo "Open http://localhost:8000 in your browser"
	npm run serve:dist

# Prepare for GitHub Pages deployment
prepare-deploy: build-pages
	@echo "Deployment preparation complete!"
	@echo "Files ready for GitHub Pages in dist/ directory"
	@echo "To deploy manually, push changes and enable GitHub Pages in repository settings"

# Size analysis
analyze:
	@echo "Analyzing WebAssembly bundle size..."
	@if [ -f pkg/multilayer_perceptron_bg.wasm ]; then \
		echo "WebAssembly bundle size:"; \
		ls -lh pkg/multilayer_perceptron_bg.wasm; \
		echo ""; \
		wasm-nm pkg/multilayer_perceptron_bg.wasm | head -20; \
	else \
		echo "WebAssembly bundle not found. Run 'make build-wasm' first."; \
	fi

# Profile build
profile:
	@echo "Building with profiling enabled..."
	CARGO_PROFILE_RELEASE_DEBUG=true wasm-pack build --target web --out-dir pkg --release

# Generate documentation
docs:
	@echo "Generating documentation..."
	cargo doc --no-deps --open

# Check WebAssembly support
check-wasm:
	@echo "Checking WebAssembly toolchain..."
	@rustup target list --installed | grep -q wasm32-unknown-unknown || (echo "Installing wasm32-unknown-unknown target..." && rustup target add wasm32-unknown-unknown)
	@command -v wasm-pack >/dev/null 2>&1 || (echo "Installing wasm-pack..." && curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh)
	@echo "WebAssembly toolchain ready!"

# Help
help:
	@echo "Available targets:"
	@echo "  install       - Install dependencies"
	@echo "  build         - Build WebAssembly modules for all targets"
	@echo "  build-wasm    - Build WebAssembly module for web"
	@echo "  build-node    - Build WebAssembly module for Node.js"
	@echo "  build-bundler - Build WebAssembly module for bundlers"
	@echo "  build-demo    - Build demo website"
	@echo "  build-pages   - Build for GitHub Pages deployment"
	@echo "  dev           - Start development server"
	@echo "  serve         - Start simple HTTP server"
	@echo "  deploy-local  - Test GitHub Pages build locally"
	@echo "  prepare-deploy- Prepare files for GitHub Pages"
	@echo "  check-wasm    - Check WebAssembly toolchain"
	@echo "  test          - Run WebAssembly tests"
	@echo "  test-chrome   - Run tests in Chrome"
	@echo "  test-rust     - Run Rust tests"
	@echo "  test-all      - Run all tests"
	@echo "  clean         - Clean build artifacts"
	@echo "  lint          - Lint Rust code"
	@echo "  format        - Format Rust code"
	@echo "  audit         - Audit dependencies for security"
	@echo "  release       - Build optimized release"
	@echo "  analyze       - Analyze WebAssembly bundle size"
	@echo "  profile       - Build with profiling"
	@echo "  docs          - Generate documentation"
	@echo "  help          - Show this help"