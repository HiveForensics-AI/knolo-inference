use std::collections::BTreeMap;

use infer_contracts::{FixedPointSamplerV1, SamplerPlanV1};
use infer_engine::{argmax, philox4x32_10, philox_u32, sample_token};

fn plan(temperature: u32, repetition: u32, seed: Option<u64>) -> SamplerPlanV1 {
    let greedy = temperature == 0;
    SamplerPlanV1 {
        settings: FixedPointSamplerV1 {
            temperature_micros: temperature,
            top_p_millionths: 1_000_000,
            min_p_millionths: 0,
            repetition_penalty_micros: repetition,
            presence_penalty_micros: 0,
            frequency_penalty_micros: 0,
            top_k: 0,
            max_output_tokens: 4,
        },
        rng: if greedy {
            "none".into()
        } else {
            "philox-4x32-v1".into()
        },
        seed: if greedy { None } else { seed },
        stream: if greedy { None } else { Some(0) },
        tie_break: "lowest-token-id".into(),
        eos_token_ids: vec![2],
        stop_string_roots: Vec::new(),
        extensions: BTreeMap::new(),
    }
}

#[test]
fn philox_matches_the_random123_known_answers() {
    assert_eq!(
        philox4x32_10([0, 0, 0, 0], [0, 0]),
        [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
    );
    assert_eq!(
        philox4x32_10([0xffff_ffff; 4], [0xffff_ffff; 2]),
        [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
    );
    assert_eq!(
        philox4x32_10(
            [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
            [0xa409_3822, 0x299f_31d0]
        ),
        [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
    );
    assert_eq!(philox_u32(0, 0, 0), 0x6627_e8d5);
}

#[test]
fn greedy_keeps_the_lowest_id_and_penalties_can_move_it() {
    let greedy = plan(0, 1_000_000, None);
    assert_eq!(sample_token(&[1.0, 3.0, 3.0], &greedy, &[], 0).unwrap(), 1);
    assert_eq!(argmax(&[1.0, 3.0, 3.0]).unwrap(), 1);
    let penalized = plan(0, 2_000_000, None);
    assert_eq!(sample_token(&[3.0, 4.0], &penalized, &[1], 0).unwrap(), 0);
}

#[test]
fn seeded_draws_repeat_for_the_same_step() {
    let seeded = plan(1_000_000, 1_000_000, Some(42));
    let logits = [0.2, 1.5, 0.1, 0.7, 0.3, 1.1];
    let first = sample_token(&logits, &seeded, &[], 0).unwrap();
    let second = sample_token(&logits, &seeded, &[], 0).unwrap();
    assert_eq!(first, second);
    let other = sample_token(&logits, &seeded, &[], 1).unwrap();
    let _ = other;
}
