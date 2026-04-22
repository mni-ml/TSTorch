//! Criterion benchmarks for the cuTile backend.
//!
//! Compares cuTile kernels against naive CPU implementations so we can report
//! a speed-up.  The CPU baseline is a single-threaded tight loop in release
//! mode; it is not meant to be a competitive baseline but provides a
//! familiar reference point.  Each bench does a warm-up pass before timing.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mni_framework_cutile::ops::{
    activation, attention, conv, elementwise, matmul, norm, reduce, softmax,
};
use mni_framework_cutile::tensor::TensorStore;
use std::hint::black_box;

fn cpu_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn cpu_saxpy(a: f32, x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(xi, yi)| a * xi + yi).collect()
}

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");
    for &n in &[1024usize, 1 << 14, 1 << 20, 1 << 22] {
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.125).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25).collect();
        group.throughput(Throughput::Elements(n as u64));

        // cuTile path — allocate once, reuse the IDs each iteration.
        {
            let mut store = TensorStore::new();
            let ia = store.from_slice(&a, &[n]);
            let ib = store.from_slice(&b, &[n]);
            // Warm up: triggers PTX compile + cache.
            let _ = elementwise::add(&mut store, ia, ib);
            group.bench_with_input(BenchmarkId::new("cutile", n), &n, |bch, _| {
                bch.iter(|| {
                    let id = elementwise::add(&mut store, ia, ib);
                    black_box(id);
                    store.free(id);
                });
            });
        }

        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bch, _| {
            bch.iter(|| black_box(cpu_add(black_box(&a), black_box(&b))));
        });
    }
    group.finish();
}

fn bench_saxpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("saxpy");
    for &n in &[1024usize, 1 << 14, 1 << 20, 1 << 22] {
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.125).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        group.throughput(Throughput::Elements(n as u64));

        {
            let mut store = TensorStore::new();
            let ix = store.from_slice(&x, &[n]);
            let iy = store.from_slice(&y, &[n]);
            let _ = elementwise::saxpy(&mut store, 2.5, ix, iy); // warm up
            group.bench_with_input(BenchmarkId::new("cutile", n), &n, |bch, _| {
                bch.iter(|| {
                    let id = elementwise::saxpy(&mut store, 2.5, ix, iy);
                    black_box(id);
                    store.free(id);
                });
            });
        }

        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bch, _| {
            bch.iter(|| black_box(cpu_saxpy(2.5, black_box(&x), black_box(&y))));
        });
    }
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    // Square matmuls — (M=N=K).  Sizes chosen to be divisible by our tile
    // picker (all multiples of 64).
    for &sz in &[64usize, 128, 256, 512] {
        let m = sz;
        let n = sz;
        let k = sz;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.07).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.05).collect();
        let flops = 2.0 * (m * n * k) as f64;
        group.throughput(Throughput::Elements((m * n * k) as u64));

        {
            let mut store = TensorStore::new();
            let ia = store.from_slice(&a, &[m, k]);
            let ib = store.from_slice(&b, &[k, n]);
            let _ = matmul::matmul(&mut store, ia, ib); // warm up
            group.bench_with_input(
                BenchmarkId::new("cutile", format!("{m}x{n}x{k}")),
                &sz,
                |bch, _| {
                    bch.iter(|| {
                        let id = matmul::matmul(&mut store, ia, ib);
                        black_box(id);
                        store.free(id);
                    });
                },
            );
        }

        // CPU baseline only for smaller sizes — it gets prohibitive beyond 512.
        if sz <= 256 {
            group.bench_with_input(
                BenchmarkId::new("cpu", format!("{m}x{n}x{k}")),
                &sz,
                |bch, _| {
                    bch.iter(|| {
                        black_box(cpu_matmul(
                            black_box(&a),
                            black_box(&b),
                            black_box(m),
                            black_box(n),
                            black_box(k),
                        ))
                    });
                },
            );
        }

        eprintln!("matmul {m}x{n}x{k}: {:.2} GFLOPs per call", flops / 1e9);
    }
    group.finish();
}

fn bench_sum_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_all");
    for &n in &[1 << 14usize, 1 << 20, 1 << 22] {
        let a: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) * 0.01).collect();
        group.throughput(Throughput::Elements(n as u64));

        {
            let mut store = TensorStore::new();
            let ia = store.from_slice(&a, &[n]);
            let warm = reduce::sum_all(&mut store, ia);
            store.free(warm);
            group.bench_with_input(BenchmarkId::new("cutile", n), &n, |bch, _| {
                bch.iter(|| {
                    let id = reduce::sum_all(&mut store, ia);
                    black_box(id);
                    store.free(id);
                });
            });
        }

        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bch, _| {
            bch.iter(|| {
                let s: f32 = black_box(&a).iter().copied().sum();
                black_box(s)
            });
        });
    }
    group.finish();
}

fn cpu_gelu(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = c * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn cpu_softmax_rowwise(x: &[f32], n: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * c];
    for r in 0..n {
        let row = &x[r * c..(r + 1) * c];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        let mut tmp = vec![0.0f32; c];
        for (j, &v) in row.iter().enumerate() {
            let e = (v - m).exp();
            tmp[j] = e;
            denom += e;
        }
        for j in 0..c {
            out[r * c + j] = tmp[j] / denom;
        }
    }
    out
}

fn bench_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gelu");
    for &n in &[1024usize, 1 << 14, 1 << 20, 1 << 22] {
        let x: Vec<f32> = (0..n).map(|i| ((i % 257) as f32) * 0.01 - 1.0).collect();
        group.throughput(Throughput::Elements(n as u64));

        {
            let mut store = TensorStore::new();
            let ix = store.from_slice(&x, &[n]);
            let warm = activation::gelu(&mut store, ix); // PTX cache warm-up.
            store.free(warm);
            group.bench_with_input(BenchmarkId::new("cutile", n), &n, |bch, _| {
                bch.iter(|| {
                    let id = activation::gelu(&mut store, ix);
                    black_box(id);
                    store.free(id);
                });
            });
        }

        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bch, _| {
            bch.iter(|| {
                let out: Vec<f32> = black_box(&x).iter().map(|&v| cpu_gelu(v)).collect();
                black_box(out);
            });
        });
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");
    // (n_rows, c).  Softmax is along dim=1 (c).
    for &(n, c_dim) in &[(64usize, 256usize), (256, 1024), (1024, 1024), (4096, 1024)] {
        let total = n * c_dim;
        let x: Vec<f32> = (0..total).map(|i| ((i % 17) as f32) * 0.05 - 0.4).collect();
        group.throughput(Throughput::Elements(total as u64));

        {
            let mut store = TensorStore::new();
            let ix = store.from_slice(&x, &[n, c_dim]);
            let warm = softmax::softmax_forward(&mut store, ix, 1);
            store.free(warm);
            group.bench_with_input(
                BenchmarkId::new("cutile", format!("{n}x{c_dim}")),
                &total,
                |bch, _| {
                    bch.iter(|| {
                        let id = softmax::softmax_forward(&mut store, ix, 1);
                        black_box(id);
                        store.free(id);
                    });
                },
            );
        }

        // CPU baseline only for the smaller shapes — gets too slow otherwise.
        if total <= 1 << 18 {
            group.bench_with_input(
                BenchmarkId::new("cpu", format!("{n}x{c_dim}")),
                &total,
                |bch, _| {
                    bch.iter(|| {
                        black_box(cpu_softmax_rowwise(black_box(&x), n, c_dim));
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_layernorm_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("layernorm_forward");
    for &(n, c_dim) in &[(64usize, 256usize), (256, 1024), (1024, 1024), (4096, 1024)] {
        let total = n * c_dim;
        let x: Vec<f32> = (0..total).map(|i| ((i % 19) as f32) * 0.02 - 0.2).collect();
        let g: Vec<f32> = (0..c_dim).map(|i| 1.0 + (i as f32) * 0.001).collect();
        let bv: Vec<f32> = (0..c_dim).map(|i| (i as f32) * 0.0005).collect();
        group.throughput(Throughput::Elements(total as u64));

        let mut store = TensorStore::new();
        let ix = store.from_slice(&x, &[n, c_dim]);
        let ig = store.from_slice(&g, &[c_dim]);
        let ib = store.from_slice(&bv, &[c_dim]);
        // Warm up — frees all three result tensors.
        let warm = norm::layernorm_forward(&mut store, ix, ig, ib, 1e-5);
        store.free(warm.out);
        store.free(warm.mean);
        store.free(warm.rstd);
        group.bench_with_input(
            BenchmarkId::new("cutile", format!("{n}x{c_dim}")),
            &total,
            |bch, _| {
                bch.iter(|| {
                    let st = norm::layernorm_forward(&mut store, ix, ig, ib, 1e-5);
                    black_box(&st);
                    store.free(st.out);
                    store.free(st.mean);
                    store.free(st.rstd);
                });
            },
        );
    }
    group.finish();
}

fn bench_flash_attention_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("flash_attention_forward");
    // (BH, S, D) — D capped to the supported set {32, 64, 96, 128}; S kept
    // as a multiple of the BM picker so the kernel grid is full.
    for &(bh, s, d) in &[(4usize, 64usize, 64usize), (4, 256, 64), (8, 512, 64)] {
        let total = bh * s * d;
        let scale = 1.0 / (d as f32).sqrt();
        let q: Vec<f32> = (0..total).map(|i| ((i % 13) as f32) * 0.01 - 0.05).collect();
        let k: Vec<f32> = (0..total).map(|i| ((i % 11) as f32) * 0.012 - 0.05).collect();
        let v: Vec<f32> = (0..total).map(|i| ((i % 17) as f32) * 0.008 - 0.05).collect();
        group.throughput(Throughput::Elements(total as u64));

        let mut store = TensorStore::new();
        let iq = store.from_slice(&q, &[bh, s, d]);
        let ik = store.from_slice(&k, &[bh, s, d]);
        let iv = store.from_slice(&v, &[bh, s, d]);
        let warm = attention::flash_attention_forward(&mut store, iq, ik, iv, scale, false);
        store.free(warm.out);
        store.free(warm.lse);
        group.bench_with_input(
            BenchmarkId::new("cutile", format!("bh{bh}_s{s}_d{d}")),
            &total,
            |bch, _| {
                bch.iter(|| {
                    let st =
                        attention::flash_attention_forward(&mut store, iq, ik, iv, scale, false);
                    black_box(&st);
                    store.free(st.out);
                    store.free(st.lse);
                });
            },
        );
    }
    group.finish();
}

fn bench_conv2d_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_forward");
    // (N, C_in, H, W, C_out, K).  Square spatial + square kernel.
    for &(n, c_in, h, w, c_out, k) in
        &[(2usize, 16usize, 32usize, 32usize, 32usize, 3usize), (2, 32, 64, 64, 64, 3)]
    {
        let inp_total = n * c_in * h * w;
        let w_total = c_out * c_in * k * k;
        let x: Vec<f32> = (0..inp_total).map(|i| ((i % 23) as f32) * 0.01 - 0.1).collect();
        let wt: Vec<f32> = (0..w_total).map(|i| ((i % 31) as f32) * 0.005 - 0.075).collect();
        // Throughput is per-output-element: N·C_out·H_out·W_out.
        let h_out = (h + 2 * 1 - k) / 1 + 1;
        let w_out = (w + 2 * 1 - k) / 1 + 1;
        let out_total = n * c_out * h_out * w_out;
        group.throughput(Throughput::Elements(out_total as u64));

        let mut store = TensorStore::new();
        let ix = store.from_slice(&x, &[n, c_in, h, w]);
        let iw = store.from_slice(&wt, &[c_out, c_in, k, k]);
        let warm = conv::conv2d_forward(&mut store, ix, iw, 1, 1);
        store.free(warm);
        group.bench_with_input(
            BenchmarkId::new("cutile", format!("n{n}_ci{c_in}_co{c_out}_{h}x{w}_k{k}")),
            &out_total,
            |bch, _| {
                bch.iter(|| {
                    let id = conv::conv2d_forward(&mut store, ix, iw, 1, 1);
                    black_box(id);
                    store.free(id);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_add,
    bench_saxpy,
    bench_matmul,
    bench_sum_all,
    bench_gelu,
    bench_softmax,
    bench_layernorm_forward,
    bench_flash_attention_forward,
    bench_conv2d_forward,
);
criterion_main!(benches);
