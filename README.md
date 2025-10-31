# Multilayer Perceptron WebAssembly Demo

🧠 Interactive neural network training in your browser using Rust + WebAssembly

## 🌐 Live Demo

Visit the live demo: [https://nktr-cp.github.io/multilayer-perceptron/](https://nktr-cp.github.io/multilayer-perceptron/)

## ✨ Features

- **Clean Architecture Core**: domain/usecase/adapters layers keep business logic decoupled from delivery concerns.
- **Task-aware Training Pipeline**: switch between binary classification, multi-class classification, and regression via a simple `TaskKind` enum; default losses/metrics/activations are selected automatically.
- **Composable Preprocessing**: build transform pipelines (standardisation, normalisation, …) that can be fitted on the training split and reused on validation/test data.
- **Interactive Neural Network Training**: Train a multilayer perceptron directly in your browser with real-time feedback.
- **WebAssembly Performance**: High-performance neural network computation using Rust and WebAssembly with a responsive UI.

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

## 🧭 Task Configuration & Preprocessing

- Select the learning objective with `domain::types::TaskKind` (`BinaryClassification`, `MultiClassification`, `Regression`).
- Leave `TrainRequest.loss_fn`, `train_metrics`, or `val_metrics` empty to let the use case choose sensible defaults (e.g. BCE + accuracy/precision/recall/F1 for binary classification, MSE for regression).
- Assemble preprocessing steps with `usecase::preprocess::build_pipeline` and fit them on the training split before applying them to validation/test data.

```rust
use multilayer_perceptron::prelude::*;
use multilayer_perceptron::usecase::preprocess::build_pipeline;
use multilayer_perceptron::usecase::{TrainMLPUsecase, TrainRequest};

let data_config = DataConfig::default();
let training_config = TrainingConfig {
    epochs: 100,
    batch_size: 32,
    ..Default::default()
};

let mut request = TrainRequest {
    task: TaskKind::MultiClassification,
    data_config: data_config.clone(),
    training_config,
    validation_split: Some(0.2),
    model: Sequential::new()
      .relu_layer(64, 32)
      .softmax_layer(32, 10),
    optimizer: Box::new(SGD::new(0.01)),
    loss_fn: None,               // auto-selected
    train_metrics: Vec::new(),    // auto-selected
    val_metrics: Vec::new(),      // auto-selected
};

let mut pipeline = build_pipeline(&request.data_config);
let mut train_usecase = TrainMLPUsecase::new(data_repo.clone());
let response = train_usecase.execute(request)?;
```

`TrainMLPUsecase` will adjust the model's output activation, choose a loss/metric bundle, and ensure the preprocessing pipeline is applied consistently across training and validation datasets.

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
src/
├── core/          # tensor, graph, ops primitives (pure math)
├── domain/        # business rules: MLP model, losses, metrics, ports, TaskKind
├── usecase/       # application services (training, preprocessing, inference)
├── adapters/      # external implementations (CSV loader, WASM bindings, native presentation)
├── app/           # high-level façade / integration glue
├── bin/           # CLI entrypoints
└── lib.rs         # crate exports & prelude

www/               # Web front-end assets
.github/workflows/ # CI pipelines
```

### Clean Architecture Flow

```
            adapters ─┐
                      ▼
bin/app ─▶ usecase ─▶ domain ─▶ core
                      ▲
                      └── ports (traits)
```

Each outer layer depends only on the layer directly beneath it. Adapters implement the `domain::ports` traits, while use cases orchestrate datasets, preprocessing pipelines, optimisers, and task-aware strategy selection.

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
