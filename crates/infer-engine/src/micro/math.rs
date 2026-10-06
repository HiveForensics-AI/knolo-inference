//! Plain f32 primitives. These loops are the numerical authority for `knolo.micro.v1`.

use infer_contracts::{fail, ErrorCode, InferFailure};

pub(crate) fn non_finite(what: &str) -> InferFailure {
    fail(
        ErrorCode::ContractInvalid,
        format!("micro forward produced a non-finite {what}"),
    )
}

pub(crate) fn gemv(
    weight: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
) -> Result<Vec<f32>, InferFailure> {
    if weight.len() != rows * cols || x.len() != cols || rows == 0 || cols == 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "micro matrix shape does not match its vector",
        ));
    }
    let mut y = vec![0.0; rows];
    for (row, slot) in y.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        let base = row * cols;
        for (col, input) in x.iter().enumerate() {
            acc += weight[base + col] * input;
        }
        if !acc.is_finite() {
            return Err(non_finite("matrix product"));
        }
        *slot = acc;
    }
    Ok(y)
}

pub(crate) fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>, InferFailure> {
    if x.is_empty() || x.len() != weight.len() {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "rmsnorm shape does not match its weight",
        ));
    }
    let mut sum = 0.0f32;
    for value in x {
        sum += value * value;
    }
    let scale = (sum / x.len() as f32 + eps).sqrt().recip();
    if !scale.is_finite() {
        return Err(non_finite("rmsnorm"));
    }
    let mut y = Vec::with_capacity(x.len());
    for (value, scale_w) in x.iter().zip(weight) {
        let out = value * scale * scale_w;
        if !out.is_finite() {
            return Err(non_finite("rmsnorm"));
        }
        y.push(out);
    }
    Ok(y)
}

/// Llama rotate-half RoPE. `theta` is the fixed base, 10000 for this model.
/// Pair `i` uses `angle = position * theta^(-2i / dim)`.
pub(crate) fn rope(head: &mut [f32], position: u32, theta: f32) -> Result<(), InferFailure> {
    if head.is_empty() || head.len() % 2 != 0 {
        return Err(fail(
            ErrorCode::ContractInvalid,
            "rope head dimension must be even",
        ));
    }
    let half = head.len() / 2;
    let dim = head.len() as f32;
    for i in 0..half {
        let inv = theta.powf(-((2 * i) as f32) / dim);
        let angle = position as f32 * inv;
        let (sin, cos) = angle.sin_cos();
        let x1 = head[i];
        let x2 = head[half + i];
        head[i] = x1 * cos - x2 * sin;
        head[half + i] = x2 * cos + x1 * sin;
        if !head[i].is_finite() || !head[half + i].is_finite() {
            return Err(non_finite("rope"));
        }
    }
    Ok(())
}

pub(crate) fn silu(x: f32) -> Result<f32, InferFailure> {
    let sigmoid = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    };
    let y = x * sigmoid;
    if !y.is_finite() {
        return Err(non_finite("silu"));
    }
    Ok(y)
}

pub(crate) fn softmax(scores: &mut [f32]) -> Result<(), InferFailure> {
    if scores.is_empty() {
        return Err(fail(ErrorCode::ContractInvalid, "softmax is empty"));
    }
    let mut max = f32::NEG_INFINITY;
    for value in scores.iter() {
        if !value.is_finite() {
            return Err(non_finite("attention score"));
        }
        max = max.max(*value);
    }
    let mut sum = 0.0f32;
    for value in scores.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum == 0.0 || !sum.is_finite() {
        return Err(non_finite("softmax"));
    }
    for value in scores.iter_mut() {
        *value /= sum;
    }
    Ok(())
}

pub(crate) fn add_residual(dst: &mut [f32], src: &[f32]) -> Result<(), InferFailure> {
    if dst.len() != src.len() {
        return Err(fail(ErrorCode::ContractInvalid, "residual shapes differ"));
    }
    for (slot, value) in dst.iter_mut().zip(src) {
        *slot += value;
        if !slot.is_finite() {
            return Err(non_finite("residual"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_at_position_zero_is_identity() {
        let mut head = vec![0.5, -0.25, 0.125, -1.0];
        let original = head.clone();
        rope(&mut head, 0, 10_000.0).unwrap();
        assert_eq!(head, original);
    }

    #[test]
    fn rmsnorm_scales_every_element_by_the_same_factor() {
        let x = [1.0f32, 2.0, 2.0, 4.0];
        let weight = [1.0, 1.0, 1.0, 2.0];
        let y = rmsnorm(&x, &weight, 1e-5).unwrap();
        let ratio = y[0] / x[0];
        assert!((y[1] / x[1] - ratio).abs() < 1e-6);
        assert!((y[2] / x[2] - ratio).abs() < 1e-6);
        assert!((y[3] / x[3] - ratio * 2.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one_and_keeps_the_largest_input() {
        let mut scores = vec![1.0, 3.0, 2.0];
        softmax(&mut scores).unwrap();
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(scores[1] > scores[0] && scores[1] > scores[2]);
    }
}
