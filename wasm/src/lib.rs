use wasm_bindgen::prelude::*;
#[wasm_bindgen]
pub fn matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m*n];
    for i in 0..m {
        for p in 0..k {
            let aval = a[i*k + p];
            let bp = p*n;
            let oi = i*n;
            for j in 0..n {
                out[oi + j] += aval * b[bp + j];
            }
        }
    }
    out
}
#[wasm_bindgen]
pub fn matmul_i8_dequant(a_i8: &[i8], a_scale: f32, b_i8: &[i8], b_scale: f32, m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m*n];
    let _s = a_scale * b_scale;
    for i in 0..m {
        for p in 0..k {
            let aval = a_i8[i*k + p] as f32 * a_scale;
            let bp = p*n;
            let oi = i*n;
            for j in 0..n {
                out[oi + j] += aval * (b_i8[bp + j] as f32 * b_scale);
            }
        }
    }
    out
}
#[wasm_bindgen]
pub fn softmax_inplace(x: &mut [f32]) {
    let mut maxv = f32::NEG_INFINITY;
    for &v in x.iter() { if v > maxv { maxv = v; } }
    let mut sum = 0.0f32;
    for v in x.iter_mut() { *v = (*v - maxv).exp(); sum += *v; }
    for v in x.iter_mut() { *v /= sum; }
}
