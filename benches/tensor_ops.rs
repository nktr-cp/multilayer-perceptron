use criterion::{black_box, criterion_group, criterion_main, Criterion};
use multilayer_perceptron::add;

fn benchmark_add(c: &mut Criterion) {
    c.bench_function("add small numbers", |b| {
        b.iter(|| add(black_box(2), black_box(2)))
    });

    c.bench_function("add large numbers", |b| {
        b.iter(|| add(black_box(1_000_000), black_box(2_000_000)))
    });
}

// Placeholder benchmarks - will be replaced with actual tensor operations
fn benchmark_tensor_operations(c: &mut Criterion) {
    // TODO: Add tensor benchmarks when tensor module is implemented
    c.bench_function("placeholder tensor operation", |b| {
        b.iter(|| {
            // Placeholder computation
            let _result = black_box(42);
        })
    });
}

criterion_group!(benches, benchmark_add, benchmark_tensor_operations);
criterion_main!(benches);
