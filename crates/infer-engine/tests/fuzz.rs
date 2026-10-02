use std::collections::BTreeMap;
use std::fs;

use infer_contracts::{fail, sha256_prefixed, ErrorCode, InferFailure, FUZZ_TARGETS};
use infer_engine::{
    fuzz_mutations, measure_corruption_fuzz, verify_corruption_fuzz, write_fuzz_report,
    CorruptionObservation, CorruptionProbe, FuzzSeed, MAX_FUZZ_SEED_BYTES,
};

struct ExactSeeds {
    seeds: BTreeMap<String, Vec<u8>>,
    identity: Identity,
}

enum Identity {
    RejectMutation,
    EchoBytes,
    AlwaysSame,
    RejectAll,
}

impl CorruptionProbe for ExactSeeds {
    fn identify(&self, target: &str, bytes: &[u8]) -> Result<Vec<u8>, InferFailure> {
        match self.identity {
            Identity::RejectAll => Err(fail(ErrorCode::ContractInvalid, "closed")),
            Identity::AlwaysSame => Ok(b"same".to_vec()),
            Identity::EchoBytes => Ok(bytes.to_vec()),
            Identity::RejectMutation => {
                let seed = self.seeds.get(target).expect("target");
                if bytes == seed.as_slice() {
                    Ok(seed.clone())
                } else {
                    Err(fail(ErrorCode::ContractInvalid, "mutated"))
                }
            }
        }
    }
}

fn seeds() -> Vec<FuzzSeed> {
    FUZZ_TARGETS
        .iter()
        .map(|target| FuzzSeed {
            target: (*target).to_string(),
            label: (*target).to_string(),
            bytes: format!("seed-{target}-0123456789").into_bytes(),
        })
        .collect()
}

fn observation() -> CorruptionObservation {
    CorruptionObservation {
        engine_build_root: sha256_prefixed(b"fuzz-engine"),
        seeds: seeds(),
    }
}

fn probe(identity: Identity) -> ExactSeeds {
    ExactSeeds {
        seeds: seeds()
            .into_iter()
            .map(|seed| (seed.target, seed.bytes))
            .collect(),
        identity,
    }
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-fuzz-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn rejected_mutations_record_a_fail_closed_report() {
    let observed = observation();
    let closed = probe(Identity::RejectMutation);
    assert_eq!(fuzz_mutations(&observed.seeds[0].bytes).len(), 6);
    let measured = measure_corruption_fuzz(&observed, &closed).unwrap();
    assert_eq!(measured.report.validation_result, "fail-closed");
    assert_eq!(measured.report.target_count, 9);
    assert_eq!(measured.report.seed_count, 9);
    assert_eq!(measured.report.mutation_count, 54);
    assert_eq!(measured.report.rejected_count, 54);
    assert_eq!(measured.report.distinguished_count, 0);
    assert_eq!(measured.report.accepted_count, 0);
    verify_corruption_fuzz(&measured, &closed).unwrap();

    let echoed = measure_corruption_fuzz(&observed, &probe(Identity::EchoBytes)).unwrap();
    assert_eq!(echoed.report.rejected_count, 0);
    assert_eq!(echoed.report.distinguished_count, 54);
}

#[test]
fn an_accepted_mutation_issues_no_report() {
    let observed = observation();
    let err = measure_corruption_fuzz(&observed, &probe(Identity::AlwaysSame)).unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message.contains("corruption was accepted for cbor"),
        "{err}"
    );

    let err = measure_corruption_fuzz(&observed, &probe(Identity::RejectAll)).unwrap_err();
    assert!(
        err.message.contains("fuzz seed was rejected for cbor"),
        "{err}"
    );

    let mut short = observation();
    short.seeds.pop();
    let err = measure_corruption_fuzz(&short, &probe(Identity::RejectMutation)).unwrap_err();
    assert!(
        err.message
            .contains("fuzz campaign is not the nine-target slice"),
        "{err}"
    );

    let mut swapped = observation();
    swapped.seeds.swap(0, 1);
    let err = measure_corruption_fuzz(&swapped, &probe(Identity::RejectMutation)).unwrap_err();
    assert!(
        err.message
            .contains("field target has an unsupported value")
            || err
                .message
                .contains("fuzz targets are not the nine-target campaign"),
        "{err}"
    );

    let mut labeled = observation();
    labeled.seeds[0].label = "other".into();
    let err = measure_corruption_fuzz(&labeled, &probe(Identity::RejectMutation)).unwrap_err();
    assert!(
        err.message.contains("fuzz label does not match the target"),
        "{err}"
    );

    let mut empty = observation();
    empty.seeds[0].bytes.clear();
    let err = measure_corruption_fuzz(&empty, &probe(Identity::RejectMutation)).unwrap_err();
    assert!(err.message.contains("fuzz seed is empty"), "{err}");

    let mut huge = observation();
    huge.seeds[0].bytes = vec![1; MAX_FUZZ_SEED_BYTES + 1];
    let err = measure_corruption_fuzz(&huge, &probe(Identity::RejectMutation)).unwrap_err();
    assert_eq!(err.code, ErrorCode::InsufficientMemory, "{err}");
    assert!(
        err.message.contains("fuzz seed exceeds the campaign bound"),
        "{err}"
    );

    let mut unknown = observation();
    unknown.seeds[0].target = "cuda-graph".into();
    let err = measure_corruption_fuzz(&unknown, &probe(Identity::RejectMutation)).unwrap_err();
    assert!(
        err.message
            .contains("field target has an unsupported value"),
        "{err}"
    );
}

#[test]
fn the_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    let observed = observation();
    let closed = probe(Identity::RejectMutation);
    let measured = measure_corruption_fuzz(&observed, &closed).unwrap();
    write_fuzz_report(&dir, &measured, &closed, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        infer_contracts::decode_contract(&stored).unwrap().kind(),
        "knolo.infer.fuzz-report"
    );
    let again = write_fuzz_report(&dir, &measured, &closed, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-fuzz-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err =
        write_fuzz_report(&dir, &measured, &closed, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());

    std::os::unix::fs::symlink(&outside, dir.join("linked.cbor")).unwrap();
    let linked = write_fuzz_report(&dir, &measured, &closed, "linked.cbor").unwrap_err();
    assert!(linked.message.contains("already exists"), "{linked}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
