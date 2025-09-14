use arrow::array::{BooleanArray, Float32Array, Float64Array};
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion_common::utils::nan_mask::build_nan_mask;

fn manual_nan_mask_f32(arr: &Float32Array) -> BooleanArray {
    BooleanArray::from_iter(arr.iter().map(|v| v.map(|x| x.is_nan())))
}

fn manual_nan_mask_f64(arr: &Float64Array) -> BooleanArray {
    BooleanArray::from_iter(arr.iter().map(|v| v.map(|x| x.is_nan())))
}

fn bench_nan_mask_f32(c: &mut Criterion) {
    let values: Vec<f32> = (0..100_000)
        .map(|i| if i % 10 == 0 { f32::NAN } else { i as f32 })
        .collect();
    let arr = Float32Array::from(values);
    let mut group = c.benchmark_group("nan_mask_f32");
    group.sample_size(10);
    group.bench_function("manual", |b| b.iter(|| manual_nan_mask_f32(&arr)));
    group.bench_function("compute", |b| b.iter(|| build_nan_mask(&arr)));
    group.finish();
}

fn bench_nan_mask_f64(c: &mut Criterion) {
    let values: Vec<f64> = (0..100_000)
        .map(|i| if i % 10 == 0 { f64::NAN } else { i as f64 })
        .collect();
    let arr = Float64Array::from(values);
    let mut group = c.benchmark_group("nan_mask_f64");
    group.sample_size(10);
    group.bench_function("manual", |b| b.iter(|| manual_nan_mask_f64(&arr)));
    group.bench_function("compute", |b| b.iter(|| build_nan_mask(&arr)));
    group.finish();
}

criterion_group!(benches, bench_nan_mask_f32, bench_nan_mask_f64);
criterion_main!(benches);
