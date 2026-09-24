# KIP-INFER-0016 — Crash recovery

Status: `knolo-infer serve` owns one home. A second process that finds the owner alive does not listen. A lock left by a dead process is taken, and a journal that stopped after `accepted` is sealed `failed` before the new process accepts work. `knolo-infer run` stays one isolated sequence. There is still no CUDA kernel. Prefix cache stays off. The model pin is still `knolo.infer.lock.json`.

## Daemon lock

The lock is the directory `{home}/daemon.lock`, mode `0700`. Its `owner` file is mode `0600` and holds two lines:

```text
pid 12345
starttime 67890
```

`pid` is the serve process. `starttime` is field 22 of `/proc/<pid>/stat`, the start time recorded by the kernel. The file does not contain a request id, a prompt, an alias, or a path. `knolo.infer.lock.json` is unchanged. It still pins the model.

The lock is taken before the listener binds and before a worker starts. A second `knolo-infer serve` on that home, while the recorded pid is alive and the start time matches, fails with `CONTRACT_INVALID` and the message `daemon lock is held`. That process does not bind, does not start a worker, does not seal journals, and does not count a restart. The process that holds the lock keeps serving. The same completion still returns the same token ids.

A lock is stale when the pid is not alive, or when the pid is alive and the start time differs. The new process replaces that directory with its own owner. An owner file that is not those two lines is stale. A symlink at `daemon.lock` is replaced and is not followed.

Shutdown removes the directory only when the owner file is this process. A killed process leaves the directory in place.

## Open journals

Recovery runs after this process owns the lock and before it listens.

A journal that starts at `accepted` and does not end at `completed`, `failed`, or `cancelled` gets one appended `failed` event. The payload root is the `infer-execution-trace` digest of the text `WORKER_LOST`. The chain is otherwise left as it was. No receipt is stored. The request id stays reserved, so a later completion with that id is `RECEIPT_PERSIST_FAILED`.

A second start does not append another `failed` event. A journal that already ends at `completed`, `failed`, or `cancelled` is not rewritten.

A request directory with no event files was never accepted. It is removed, and that request id can be used. A symlink inside `journals/`, a journal name that is not a request id, or a chain that does not verify stays on disk. The start fails with that chain's code, and the daemon lock is not held afterward.

`knolo_infer_worker_restarts_total` on the new process starts at 0. Taking a stale lock is not a worker restart. A ready worker that exits inside the new process still counts one restart, as KIP-INFER-0010 specifies.

After recovery, a new request id returns the same token ids as that request run alone. The listener stays up.

## Out of this slice

Authentication and CUDA stay out. Prefix cache stays off. Recovery does not delete a terminal journal, does not store a receipt for the sealed failure, and does not cap journal or trace disk. `knolo-infer run` does not take this lock.
