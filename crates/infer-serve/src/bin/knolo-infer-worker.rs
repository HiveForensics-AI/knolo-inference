//! Worker process. It speaks the supervisor socket and does not listen.

fn main() {
    if infer_serve::worker_main(std::env::args().skip(1)).is_err() {
        std::process::exit(1);
    }
}
