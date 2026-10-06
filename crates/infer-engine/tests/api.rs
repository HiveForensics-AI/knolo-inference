use std::fs;
use std::path::PathBuf;

use infer_contracts::{decode_contract, sha256_prefixed, ErrorCode, MAX_API_BODY, MAX_API_RATE};
use infer_engine::{
    cpu_placement, load_verified_micro, measure_api_boundary, reference_engine_build,
    reference_kernel_bundle, verify_api_boundary, write_api_report, write_synthetic_model,
    ApiObservation,
};

fn engine_root() -> infer_contracts::DigestHex {
    let bundle = reference_kernel_bundle().unwrap();
    let build = reference_engine_build(sha256_prefixed(b"knolo-infer-api"), bundle.root().unwrap())
        .unwrap();
    build.root().unwrap()
}

fn pin(bytes: &[u8]) -> infer_contracts::DigestHex {
    sha256_prefixed(bytes)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("knolo-infer-api-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn observe(bind: &str, remote: bool, body: u64, rate: u32) -> ApiObservation {
    ApiObservation {
        engine_build_root: engine_root(),
        tenant_root: pin(b"tenant"),
        bind: bind.into(),
        remote_explicit: remote,
        auth: "hook".into(),
        body_limit_bytes: body,
        rate_per_minute: rate,
        concurrency_limit: 1,
        prompt_log: "omitted".into(),
        metrics_labels: "counts".into(),
        execution_mode: "pinned".into(),
        cache_policy: "off".into(),
        concurrency: 1,
        run_count: 1,
        warm_state: "cold".into(),
        request_count: 1,
    }
}

#[test]
fn the_api_boundary_records_localhost_and_an_explicit_remote_bind() {
    let dir = scratch("api");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let local = measure_api_boundary(&plan, &observe("localhost", false, 1024, 60)).unwrap();
    assert_eq!(local.report.bind, "localhost");
    assert_eq!(local.report.auth, "hook");
    assert_eq!(local.report.prompt_log, "omitted");
    assert_eq!(local.report.metrics_labels, "counts");
    verify_api_boundary(&local).unwrap();

    let slot = infer_engine::cuda_placement(&source).unwrap();
    let remote =
        measure_api_boundary(&slot, &observe("remote", true, MAX_API_BODY, MAX_API_RATE)).unwrap();
    assert!(remote.report.remote_explicit);
    assert_eq!(remote.report.body_limit_bytes, MAX_API_BODY);
    assert_eq!(remote.report.rate_per_minute, MAX_API_RATE);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_api_boundary_refuses_a_bearer_a_quiet_remote_bind_and_an_oversized_body() {
    let dir = scratch("refuse");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();

    let mut auth = observe("localhost", false, 1024, 60);
    auth.auth = "bearer".into();
    let err = measure_api_boundary(&plan, &auth).unwrap_err();
    assert!(err.message.contains("authentication stays a hook"), "{err}");

    let err = measure_api_boundary(&plan, &observe("remote", false, 1024, 60)).unwrap_err();
    assert!(err.message.contains("a remote bind is explicit"), "{err}");

    let err = measure_api_boundary(&plan, &observe("localhost", true, 1024, 60)).unwrap_err();
    assert!(
        err.message.contains("localhost does not set a remote bind"),
        "{err}"
    );

    let mut prompt = observe("localhost", false, 1024, 60);
    prompt.prompt_log = "present".into();
    let err = measure_api_boundary(&plan, &prompt).unwrap_err();
    assert!(
        err.message.contains("ordinary logs omit prompt text"),
        "{err}"
    );

    let mut labels = observe("localhost", false, 1024, 60);
    labels.metrics_labels = "user".into();
    let err = measure_api_boundary(&plan, &labels).unwrap_err();
    assert!(
        err.message.contains("metrics labels omit user content"),
        "{err}"
    );

    let err = measure_api_boundary(&plan, &observe("localhost", false, 0, 60)).unwrap_err();
    assert!(
        err.message.contains("the request body limit is zero"),
        "{err}"
    );

    let err = measure_api_boundary(&plan, &observe("localhost", false, MAX_API_BODY + 1, 60))
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::InsufficientMemory, "{err}");
    assert!(
        err.message.contains("the request body exceeds 1 MiB"),
        "{err}"
    );

    let err = measure_api_boundary(&plan, &observe("localhost", false, 1024, 0)).unwrap_err();
    assert!(err.message.contains("the api rate is zero"), "{err}");

    let err = measure_api_boundary(&plan, &observe("localhost", false, 1024, MAX_API_RATE + 1))
        .unwrap_err();
    assert!(err.message.contains("the api rate exceeds 256"), "{err}");

    let mut limit = observe("localhost", false, 1024, 60);
    limit.concurrency_limit = 2;
    let err = measure_api_boundary(&plan, &limit).unwrap_err();
    assert!(
        err.message.contains("the api concurrency limit is one"),
        "{err}"
    );

    let mut tenant = observe("localhost", false, 1024, 60);
    tenant.tenant_root = tenant.engine_build_root.clone();
    let err = measure_api_boundary(&plan, &tenant).unwrap_err();
    assert!(
        err.message.contains("the tenant repeats the engine build"),
        "{err}"
    );

    let measured = measure_api_boundary(&plan, &observe("localhost", false, 1024, 60)).unwrap();
    let mut stored = measured.report.clone();
    stored.body_limit_bytes = MAX_API_BODY + 1;
    let err = stored.validate().unwrap_err();
    assert_eq!(err.code, ErrorCode::ContractInvalid, "{err}");
    assert!(
        err.message.contains("the request body exceeds 1 MiB"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn the_api_report_is_written_once_and_a_symlink_is_not_followed() {
    let dir = scratch("write");
    write_synthetic_model(&dir).unwrap();
    let source = load_verified_micro(&dir.join("micro.kmodel"), &dir).unwrap();
    let plan = cpu_placement(&source).unwrap();
    let measured = measure_api_boundary(&plan, &observe("localhost", false, 1024, 60)).unwrap();
    write_api_report(&dir, &measured, "report.cbor").unwrap();
    let stored = fs::read(dir.join("report.cbor")).unwrap();
    assert_eq!(stored, measured.report.to_bytes().unwrap());
    assert_eq!(
        decode_contract(&stored).unwrap().kind(),
        "knolo.infer.api-report"
    );
    let again = write_api_report(&dir, &measured, "report.cbor").unwrap_err();
    assert!(again.message.contains("already exists"), "{again}");

    let outside_name = format!("knolo-api-not-created-{}", std::process::id());
    let outside = std::env::temp_dir().join(&outside_name);
    let _ = fs::remove_file(&outside);
    std::os::unix::fs::symlink(std::env::temp_dir(), dir.join("escape")).unwrap();
    let err = write_api_report(&dir, &measured, &format!("escape/{outside_name}")).unwrap_err();
    assert!(err.message.contains("leaves the directory"), "{err}");
    assert!(!outside.exists());
    let _ = fs::remove_dir_all(&dir);
}
