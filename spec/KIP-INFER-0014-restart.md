# KIP-INFER-0014 — Restart

Status: `knolo-infer serve` can load the pinned model again after unload. The listener stays up. The call uses the pin this process already resolved. It does not take a new alias, a new path, or a download. `knolo-infer run` stays one isolated sequence. There is still no CUDA kernel. The worker does not restart itself.

## Endpoint

```text
POST /knolo/infer/v1/models/load
```

The body is `{}`. Unknown fields fail with `CONTRACT_INVALID` and do not change the lifecycle. Any other method is 405. The call is idempotent. It does not stop the listener and does not count a restart.

The response is 200:

```json
{"inflight": 0, "status": "serving"}
```

`inflight` counts completions that are reserved or already admitted. `status` is the lifecycle after the call. A second call reports that lifecycle. It does not replace a worker that is already ready.

## Admission

Load leaves a ready worker running. An in-flight completion keeps its token ids. Cancel still addresses that request.

When unload has been requested and `inflight` is not 0, load does not cancel that work and does not start a second worker. The response status is `unloading`. The worker exits when that work reaches `inflight` 0, as unload specifies. A later load is what starts the pin again.

When the worker is down and unload is not waiting on admitted work, load starts that same pin. The new worker has a fresh page pool. The pages of the previous worker were zeroed on its way out. A completion after that load matches the same request run before unload. `stop` and `length` still store a receipt.

Load does not clear drain. After unload plus drain, load starts the worker and the lifecycle is `drained`. A new completion stays `SERVICE_DRAINING`. Drain after unload, without load, still does not start a worker.

A failed start leaves the lifecycle unloaded when the call had left unload. The listener stays up.

## Health

`GET /knolo/infer/v1/health` stays 200 and includes `lifecycle`.

| `lifecycle` | When | `worker` |
|---|---|---|
| `serving` | load cleared unload, drain is clear, and the worker is ready | `ready` |
| `drained` | drain is still set and `inflight` is 0 | `ready` after a load that found the worker down |
| `unloading` | load arrived while admitted work was still running | `ready` until that work finishes |
| `unloaded` | unload finished and load has not started a worker | `down` |

`knolo_infer_worker_restarts_total` does not increase. Unload did not count the exit, and this load does not count the start. A ready worker that exits outside unload still counts one restart. Live KV gauges follow the new worker after the next scrape.

## Process exit

SIGINT and SIGTERM still follow the drain rule: drain, wait up to 30 seconds, then shut the listener down. The HTTP load call does not exit the process.

## Out of this slice

Authentication and CUDA stay out. Prefix cache stays off. Load does not warm a second model and does not change the pin.
