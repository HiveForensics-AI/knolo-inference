use infer_contracts::ErrorCode;
use infer_engine::{micro_kv_layout, KvLayout, KvStore, PagedKv};

fn layout() -> KvLayout {
    micro_kv_layout()
}

fn code(err: infer_contracts::InferFailure) -> ErrorCode {
    err.code
}

fn write_token(kv: &mut PagedKv, sequence: u64, token: u32, mark: f32) {
    let width = layout().kv_heads as usize * layout().head_dim as usize;
    let values = vec![mark; width];
    for layer in 0..layout().layers {
        kv.write_k(sequence, layer, token, &values).unwrap();
        kv.write_v(sequence, layer, token, &values).unwrap();
    }
    kv.commit_token(sequence).unwrap();
}

#[test]
fn pages_commit_across_a_boundary_and_reuse_the_lowest_id() {
    let mut kv = PagedKv::new(layout(), 2).unwrap();
    kv.begin_sequence(1).unwrap();
    assert!(kv.block_table(1).unwrap().is_empty());
    for token in 0..16 {
        write_token(&mut kv, 1, token, 1.0);
    }
    assert_eq!(kv.block_table(1).unwrap(), vec![0]);
    assert_eq!(kv.token_len(1).unwrap(), 16);
    write_token(&mut kv, 1, 16, 4.0);
    assert_eq!(kv.block_table(1).unwrap(), vec![0, 1]);
    assert_eq!(kv.token_len(1).unwrap(), 17);

    let mut first = vec![0.0; 4];
    let mut later = vec![0.0; 4];
    kv.read_k(1, 0, 0, &mut first).unwrap();
    kv.read_k(1, 0, 16, &mut later).unwrap();
    assert_eq!(first, vec![1.0; 4]);
    assert_eq!(later, vec![4.0; 4]);

    kv.release_sequence(1).unwrap();
    kv.begin_sequence(2).unwrap();
    kv.write_k(2, 0, 0, &[9.0; 4]).unwrap();
    assert_eq!(kv.block_table(2).unwrap(), vec![0]);
    kv.read_k(2, 1, 0, &mut first).unwrap();
    assert_eq!(first, vec![0.0; 4]);
}

#[test]
fn abort_drops_the_reserved_page_and_keeps_the_committed_length() {
    let mut kv = PagedKv::new(layout(), 1).unwrap();
    kv.begin_sequence(1).unwrap();
    let values = vec![3.0; 4];
    kv.write_k(1, 0, 0, &values).unwrap();
    assert_eq!(kv.token_len(1).unwrap(), 0);
    assert_eq!(kv.block_table(1).unwrap(), vec![0]);
    kv.abort_pending(1).unwrap();
    assert!(kv.block_table(1).unwrap().is_empty());
    assert_eq!(
        code(kv.read_k(1, 0, 0, &mut [0.0; 4]).unwrap_err()),
        ErrorCode::ContractInvalid
    );
    write_token(&mut kv, 1, 0, 2.0);
    assert_eq!(kv.token_len(1).unwrap(), 1);
    assert_eq!(kv.block_table(1).unwrap(), vec![0]);
    let mut got = [0.0; 4];
    kv.read_v(1, 0, 0, &mut got).unwrap();
    assert_eq!(got, [2.0; 4]);
}

#[test]
fn invalidate_zeros_pages_and_drops_sequences() {
    let mut kv = PagedKv::new(layout(), 2).unwrap();
    kv.begin_sequence(1).unwrap();
    write_token(&mut kv, 1, 0, 7.0);
    assert_eq!(kv.census().pinned, 1);
    kv.invalidate();
    assert_eq!(kv.census().pinned, 0);
    assert_eq!(kv.census().free, kv.census().total);
    assert_eq!(
        code(kv.token_len(1).unwrap_err()),
        ErrorCode::ContractInvalid
    );
    kv.begin_sequence(2).unwrap();
    kv.write_k(2, 0, 0, &[9.0; 4]).unwrap();
    let mut other_layer = vec![0.0; 4];
    kv.read_k(2, 1, 0, &mut other_layer).unwrap();
    assert_eq!(other_layer, vec![0.0; 4]);
}

#[test]
fn active_sequences_are_not_evicted() {
    let mut kv = PagedKv::new(layout(), 2).unwrap();
    kv.begin_sequence(1).unwrap();
    kv.begin_sequence(2).unwrap();
    write_token(&mut kv, 1, 0, 1.0);
    write_token(&mut kv, 2, 0, 2.0);
    kv.deactivate(1).unwrap();
    kv.begin_sequence(3).unwrap();
    write_token(&mut kv, 3, 0, 3.0);
    assert_eq!(
        code(kv.token_len(1).unwrap_err()),
        ErrorCode::ContractInvalid
    );
    assert_eq!(kv.token_len(2).unwrap(), 1);
    assert_eq!(kv.block_table(3).unwrap(), vec![0]);

    kv.begin_sequence(4).unwrap();
    assert_eq!(
        code(kv.write_k(4, 0, 0, &[1.0; 4]).unwrap_err()),
        ErrorCode::InsufficientMemory
    );
    assert_eq!(kv.token_len(2).unwrap(), 1);
    assert_eq!(kv.token_len(4).unwrap(), 0);
}

#[test]
fn an_oversized_pool_is_refused() {
    match PagedKv::new(layout(), 100_000) {
        Ok(_) => panic!("oversized kv pool was accepted"),
        Err(err) => assert_eq!(code(err), ErrorCode::InsufficientMemory),
    }
}
