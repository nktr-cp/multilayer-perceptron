use criterion::{black_box, criterion_group, criterion_main, Criterion};
use multilayer_perceptron::add;

fn benchmark_add(c: &mut Criterion) {
    c.bench_function("add", |b| {
        b.iter(|| add(black_box(2), black_box(2)))
    });
}

criterion_group!(benches, benchmark_add);
criterion_main!(benches);
