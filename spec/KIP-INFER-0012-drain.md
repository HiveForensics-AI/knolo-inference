# KIP-INFER-0012 — Drain

Status: `knolo-infer serve` can stop admitting new completions and let admitted work finish. The worker still owns the weights, the page pool, and the scheduler, and it still does not link Candle. `knolo-infer run` stays one isolated sequence. There is still no CUDA kernel. Drain does not unload the model.

## Endpoint

```text
POST /knolo/infer/v1/drain
```

The body is `{}`. Unknown fields fail with `CONTRACT_INVALID` and do not change the lifecycle. Any other method is 405. The call is idempotent. It does not stop the listener, does not stop the worker, and does not count a restart.

The response is 200:

```json
{"inflight": 1, "status": "draining"}
```

`inflight` counts completions that are reserved or already admitted. `status` is `drained` when that count is 0, and `draining` otherwise. A second call reports the current count. It does not cancel admitted work.

## Admission

While the lifecycle is `draining` or `drained`, `POST /knolo/infer/v1/complete` and `POST /v1/chat/completions` fail with `SERVICE_DRAINING` and HTTP 503. The body is not parsed. The code is retryable. The trace is one `api` line in `{home}/traces/rejected.jsonl` with `outcome` `rejected` and `code` `SERVICE_DRAINING`. No request file is opened, and the line does not carry the body. The refusal increments `knolo_infer_admission_rejected_total{code="OTHER"}`.

A completion that reserved a slot before drain was set still runs. Its token ids match the same request run without drain. `stop` and `length` still store a receipt. Cancel still addresses that request. Health, metrics, and receipt reads still answer. A drained supervisor does not start a worker to serve a new completion.

## Health

`GET /knolo/infer/v1/health` stays 200 and includes `lifecycle`.

| `lifecycle` | When | `status` |
|---|---|---|
| `serving` | drain has not been requested | `ok` when the worker is ready, otherwise `degraded` |
| `draining` | drain was requested and `inflight` is not 0 | `draining` |
| `drained` | drain was requested and `inflight` is 0 | `drained` |

`worker` stays `ready`, `starting`, or `down`. A worker that is already ready stays ready, so the model stays loaded.

## Process exit

SIGINT and SIGTERM on `knolo-infer serve` drain, wait up to 30 seconds for admitted work, then shut the listener down. The HTTP drain call does not exit the process.

## Out of this slice

Authentication, unload, and CUDA stay out. Prefix cache stays off. Drain does not free KV pages except by letting an admitted request finish or cancel.
