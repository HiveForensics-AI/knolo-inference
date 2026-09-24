# KIP-INFER-0013 — Unload

Status: `knolo-infer serve` can drop the loaded model after admitted work finishes. The listener stays up. A new completion does not start a worker. `knolo-infer run` stays one isolated sequence. There is still no CUDA kernel. Unload does not restart the model.

## Endpoint

```text
POST /knolo/infer/v1/models/unload
```

The body is `{}`. Unknown fields fail with `CONTRACT_INVALID` and do not change the lifecycle. Any other method is 405. The call is idempotent. It does not stop the listener and does not count a restart.

The response is 200:

```json
{"inflight": 1, "status": "unloading"}
```

`inflight` counts completions that are reserved or already admitted. `status` is `unloaded` when that count is 0, and `unloading` otherwise. A second call reports the current count. It does not cancel admitted work.

When `inflight` is 0, the supervisor closes the worker. The worker zeros every KV page, drops every sequence, and exits. The page pool does not survive the process. A later call does not start another worker.

## Admission

While the lifecycle is `unloading` or `unloaded`, `POST /knolo/infer/v1/complete` and `POST /v1/chat/completions` fail with `SERVICE_UNLOADED` and HTTP 503. The body is not parsed. The code is not retryable. The trace is one `api` line in `{home}/traces/rejected.jsonl` with `outcome` `rejected` and `code` `SERVICE_UNLOADED`. No request file is opened, and the line does not carry the body. The refusal increments `knolo_infer_admission_rejected_total{code="OTHER"}`.

A completion that reserved a slot before unload was set still runs. Its token ids match the same request run without unload. `stop` and `length` still store a receipt. Cancel still addresses that request. Health, metrics, and receipt reads still answer. When that admitted work reaches `inflight` 0, the worker exits.

Unload wins over drain. A completion refused after unload is `SERVICE_UNLOADED`, including when drain was already set. A drain call after unload does not start a worker and does not change the unload lifecycle.

## Health

`GET /knolo/infer/v1/health` stays 200 and includes `lifecycle`.

| `lifecycle` | When | `status` | `worker` |
|---|---|---|---|
| `unloading` | unload was requested and the worker has not exited | `unloading` | `ready` while admitted work is still running |
| `unloaded` | unload was requested, `inflight` is 0, and the worker has exited | `unloaded` | `down` |

`serving`, `draining`, and `drained` stay as specified when unload has not been requested. Live KV gauges are 0 once the worker is down. `knolo_infer_worker_restarts_total` does not increase.

## Process exit

SIGINT and SIGTERM still follow the drain rule: drain, wait up to 30 seconds, then shut the listener down. The HTTP unload call does not exit the process.

## Out of this slice

Restart, authentication, and CUDA stay out. Prefix cache stays off. Unload does not reload the pin.
