# Multilayer Perceptron WebAssembly Demo

🧠 Interactive neural network training in your browser using Rust + WebAssembly

## 🌐 Live Demo

Visit the live demo: [https://nktr-cp.github.io/multilayer-perceptron/](https://nktr-cp.github.io/multilayer-perceptron/)

## ✨ Features

- **Interactive Neural Network Training**: Train a multilayer perceptron directly in your browser
- **Real-time Visualization**: Watch loss curves and accuracy metrics update in real-time
- **Multiple Dataset Types**: Choose from various classification problems (XOR, circular, spiral patterns, etc.)
- **WebAssembly Performance**: High-performance neural network computation using Rust and WebAssembly
- **Responsive Design**: Works on desktop and mobile devices
- **No Backend Required**: Everything runs client-side in your browser

## 🚀 Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (1.80.0 or later)
- [Node.js](https://nodejs.org/) (18.0.0 or later)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)

### Installation

```bash
# Clone the repository
git clone https://github.com/nktr-cp/multilayer-perceptron.git
cd multilayer-perceptron

# Install dependencies
make install
# or
npm install

# Check WebAssembly toolchain
make check-wasm
```

### Development

```bash
# Start development server with hot reload
make dev
# or
npm run dev

# Build for production
make build-pages
# or
npm run build:pages

# Test locally (serves the production build)
make deploy-local
# or
npm run serve:dist
```

## 📦 Building and Deployment

### Local Development

```bash
# Development server
make dev                    # Start dev server at http://localhost:8080

# Production build
make build-pages           # Build optimized version for GitHub Pages
make deploy-local          # Test production build locally at http://localhost:8000
```

### GitHub Pages Deployment

This project is configured for automatic deployment to GitHub Pages:

1. **Automatic Deployment**: 
   - Push to `main` branch triggers automatic deployment via GitHub Actions
   - The workflow builds the optimized WebAssembly and creates static files
   - GitHub Pages serves the content from the `dist/` directory

2. **Manual Deployment**:
   ```bash
   make prepare-deploy     # Build and prepare files for deployment
   git add .
   git commit -m "Deploy to GitHub Pages"
   git push origin main    # Triggers automatic deployment
   ```

3. **Local Testing**:
   ```bash
   make deploy-local       # Test the GitHub Pages build locally
   ```

### Build Optimization

The build process includes several optimizations for WebAssembly:

- **Size Optimization**: Uses `opt-level = "s"` and `lto = true` in Cargo.toml
- **WASM Optimization**: Applies `wasm-opt -Os` for further size reduction
- **Code Splitting**: Webpack splits vendor libraries and WebAssembly modules
- **Minification**: HTML, CSS, and JavaScript are minified in production builds

## 🏗️ Project Structure

```
├── .github/workflows/
│   └── deploy.yml          # GitHub Actions workflow for deployment
├── src/                    # Rust source code
│   ├── lib.rs             # Main library
│   ├── tensor.rs          # Tensor operations
│   ├── layers.rs          # Neural network layers
│   ├── trainer.rs         # Training logic
│   └── wasm.rs            # WebAssembly bindings
├── www/                   # Web frontend
│   ├── index.html         # Main HTML file
│   ├── index.js           # JavaScript entry point
│   └── style.css          # Styles
├── pkg/                   # Generated WebAssembly package
├── dist/                  # Built static files (GitHub Pages)
├── Cargo.toml             # Rust configuration
├── package.json           # Node.js configuration
├── webpack.config.js      # Webpack build configuration
├── wasm-pack.toml         # WebAssembly optimization settings
└── Makefile               # Build automation
```

## 🧪 Testing

```bash
# Run all tests
make test-all

# Rust tests
make test-rust
cargo test

# WebAssembly tests
make test
wasm-pack test --headless --firefox

# Test in Chrome
make test-chrome
wasm-pack test --headless --chrome
```

## 🔧 Development Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make install` | Install all dependencies |
| `make dev` | Start development server |
| `make build-pages` | Build for GitHub Pages |
| `make deploy-local` | Test deployment locally |
| `make clean` | Clean all build artifacts |
| `make test-all` | Run all tests |
| `make lint` | Lint Rust code |
| `make format` | Format Rust code |
| `make analyze` | Analyze WebAssembly bundle size |

## 📊 Performance

The WebAssembly build is optimized for both size and performance:

- **Bundle Size**: ~200KB (compressed)
- **Load Time**: <2 seconds on fast connections
- **Training Speed**: Real-time training with 60 FPS visualization
- **Memory Usage**: Efficient tensor operations with minimal allocations

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT OR Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [wasm-pack](https://rustwasm.github.io/wasm-pack/) for WebAssembly tooling
- [Chart.js](https://www.chartjs.org/) for visualization
- [ndarray](https://github.com/rust-ndarray/ndarray) for tensor operations
- [Rust WebAssembly Book](https://rustwasm.github.io/docs/book/) for guidance

## 🔗 Links

- [Live Demo](https://nktr-cp.github.io/multilayer-perceptron/)
- [GitHub Repository](https://github.com/nktr-cp/multilayer-perceptron)
- [Documentation](https://docs.rs/multilayer-perceptron)
- [Issues](https://github.com/nktr-cp/multilayer-perceptron/issues)

---

Made with ❤️ using Rust 🦀 and WebAssembly 🕷️