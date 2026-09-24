//! Philox-4x32 with 10 rounds, the Random123 bijection.
//!
//! `philox-4x32-v1` maps a sampler step onto that bijection:
//! key is `(seed low, seed high)` and the counter is
//! `(step, stream low, stream high, 0)`. Step 0 is counter word 0, not 1.

const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

pub fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = counter;
    let mut k = key;
    for _ in 0..10 {
        let (hi0, lo0) = mulhilo(M0, c[0]);
        let (hi1, lo1) = mulhilo(M1, c[2]);
        c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0];
        k[0] = k[0].wrapping_add(W0);
        k[1] = k[1].wrapping_add(W1);
    }
    c
}

pub fn philox_u32(seed: u64, stream: u64, step: u32) -> u32 {
    let key = [seed as u32, (seed >> 32) as u32];
    let counter = [step, stream as u32, (stream >> 32) as u32, 0];
    philox4x32_10(counter, key)[0]
}

/// Uniform draw in `[0, 1)`.
pub fn philox_unit(seed: u64, stream: u64, step: u32) -> f64 {
    f64::from(philox_u32(seed, stream, step)) / 4_294_967_296.0
}

fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let product = u64::from(a) * u64::from(b);
    ((product >> 32) as u32, product as u32)
}
