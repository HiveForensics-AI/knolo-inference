use infer_contracts::digest_bytes;
use infer_engine::{
    cpu_kernel_bundle, cuda_engine_build, cuda_kernel_bundle, cuda_kernel_plan_root,
    host_engine_build,
};

#[test]
fn cpu_bundle_names_no_cuda_kernels() {
    let bundle = cpu_kernel_bundle("0.8.4").unwrap();
    assert_eq!(bundle.build_mode, "cpu");
    assert_eq!(bundle.cuda_toolkit_version, "none");
    assert!(bundle.cuda_architectures.is_empty());
    assert_eq!(
        bundle.source_root,
        digest_bytes("infer-kernel-bundle", b"no-cuda-kernels").unwrap()
    );
    let engine = host_engine_build(
        digest_bytes("infer-engine-build", b"binary").unwrap(),
        "0.8.4",
        bundle.root().unwrap(),
    )
    .unwrap();
    assert_eq!(engine.tensor_backend, "candle-cpu");
    assert!(engine.feature_set.is_empty());
}

#[test]
fn cuda_bundle_names_toolkit_and_architecture() {
    let bundle = cuda_kernel_bundle("0.8.4", "12.2.140", "7.5").unwrap();
    assert_eq!(bundle.build_mode, "cuda");
    assert_eq!(bundle.cuda_toolkit_version, "12.2.140");
    assert_eq!(bundle.cuda_architectures, ["7.5".to_string()]);
    assert!(bundle.compiler_flags.is_empty());
    assert!(bundle.jit.is_none());
    assert_eq!(
        bundle.source_root,
        digest_bytes("infer-kernel-bundle", b"candle-cuda-kernels").unwrap()
    );
    assert_eq!(
        bundle.code_object_root,
        digest_bytes("infer-kernel-bundle", b"candle-cuda-0.8.4").unwrap()
    );
    assert_ne!(
        bundle.source_root,
        cpu_kernel_bundle("0.8.4").unwrap().source_root
    );
    let plan = cuda_kernel_plan_root().unwrap();
    assert_eq!(
        plan,
        digest_bytes(
            "infer-kernel-bundle",
            b"candle-cuda:rmsnorm,rope,attention,swiglu,gemv"
        )
        .unwrap()
    );
    let engine = cuda_engine_build(
        digest_bytes("infer-engine-build", b"binary").unwrap(),
        "0.8.4",
        bundle.root().unwrap(),
    )
    .unwrap();
    assert_eq!(engine.tensor_backend, "candle-cuda");
    assert_eq!(engine.tensor_backend_version, "0.8.4");
    assert_eq!(engine.feature_set, ["cuda".to_string()]);
    assert!(cuda_kernel_bundle("0.8.4", "none", "7.5").is_err());
    assert!(cuda_kernel_bundle("0.8.4", "12.2.140", "").is_err());
}
