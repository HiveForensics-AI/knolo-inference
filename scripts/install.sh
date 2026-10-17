#!/usr/bin/env bash
# Install knolo-infer and knolo-infer-worker.
# When nvidia-smi is on PATH and knolo-infer-linux-x86_64-cuda.tar.gz is in the
# dist directory or the release, that archive is installed. If it is missing,
# the script says so and installs the CPU archive. AMD and Huawei machines
# stay on the CPU archive and name the ROCm or Ascend archive that is not published.
# The release workflow does not build the CUDA archive.
set -euo pipefail

repo="${KNOLO_INFER_REPO:-HiveForensics-AI/knolo-inference}"
prefix="${KNOLO_INFER_PREFIX:-${HOME}/.local}"
cpu_asset="knolo-infer-linux-x86_64.tar.gz"
cuda_asset="knolo-infer-linux-x86_64-cuda.tar.gz"
asset="${cpu_asset}"
version="${1:-latest}"

if [[ "${EUID}" -eq 0 ]]; then
  echo "knolo-infer: refuse to install as root" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "knolo-infer: python3 is required to check SHA256SUMS" >&2
  exit 1
fi

os=$(uname -s)
arch=$(uname -m)
if [[ "${os}" != "Linux" || "${arch}" != "x86_64" ]]; then
  echo "knolo-infer: ${os} ${arch} has no release archive" >&2
  exit 1
fi

dist_base() {
  if [[ -n "${KNOLO_INFER_DIST_URL:-}" ]]; then
    printf '%s' "${KNOLO_INFER_DIST_URL%/}"
  elif [[ "${version}" == "latest" ]]; then
    printf '%s' "https://github.com/${repo}/releases/latest/download"
  else
    printf '%s' "https://github.com/${repo}/releases/download/${version}"
  fi
}

# A dist directory is a file check. A release URL is a HEAD request.
# The check stays inside `if`, so a missing file does not trip `set -e`.
asset_present() {
  local name="$1"
  if [[ -n "${KNOLO_INFER_DIST_DIR:-}" ]]; then
    [[ -f "${KNOLO_INFER_DIST_DIR}/${name}" ]]
    return
  fi
  curl --fail --location --silent --show-error --head --output /dev/null "$(dist_base)/${name}"
}

# `nvidia-smi` wins when an AMD or Huawei tool is also on PATH.
if command -v nvidia-smi >/dev/null 2>&1; then
  if asset_present "${cuda_asset}"; then
    asset="${cuda_asset}"
  else
    echo "knolo-infer: CUDA archive is not published yet; installing the CPU build" >&2
  fi
elif command -v amd-smi >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; then
  echo "knolo-infer: ROCm archive is not published yet; installing the CPU build" >&2
elif command -v npu-smi >/dev/null 2>&1; then
  echo "knolo-infer: Ascend archive is not published yet; installing the CPU build" >&2
fi

work=$(mktemp -d)
cleanup() {
  rm -rf "${work}"
}
trap cleanup EXIT

fetch() {
  local name="$1"
  local dest="$2"
  if [[ -n "${KNOLO_INFER_DIST_DIR:-}" ]]; then
    cp "${KNOLO_INFER_DIST_DIR}/${name}" "${dest}"
    return
  fi
  curl --fail --location --silent --show-error --output "${dest}" "$(dist_base)/${name}"
}

fetch "${asset}" "${work}/${asset}"
fetch "SHA256SUMS" "${work}/SHA256SUMS"

python3 - "${work}" "${asset}" <<'PY'
import hashlib, pathlib, sys
work = pathlib.Path(sys.argv[1])
asset = sys.argv[2]
sums = {}
for line in (work / "SHA256SUMS").read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    digest, name = line.split(None, 1)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise SystemExit(f"knolo-infer: bad checksum line for {name}")
    sums[name] = digest
if asset not in sums:
    raise SystemExit(f"knolo-infer: SHA256SUMS has no entry for {asset}")
got = hashlib.sha256((work / asset).read_bytes()).hexdigest()
if got != sums[asset]:
    raise SystemExit("knolo-infer: archive checksum did not match")
PY

tar -xzf "${work}/${asset}" -C "${work}"
for bin in knolo-infer knolo-infer-worker; do
  if [[ ! -f "${work}/${bin}" ]]; then
    echo "knolo-infer: archive is missing ${bin}" >&2
    exit 1
  fi
done

# The archive still extracts the short names. SHA256SUMS records cpu/ or cuda/
# so one file can list both builds.
if [[ "${asset}" == "${cuda_asset}" ]]; then
  bin_prefix="cuda"
else
  bin_prefix="cpu"
fi

python3 - "${work}" "${bin_prefix}" <<'PY'
import hashlib, pathlib, sys
work = pathlib.Path(sys.argv[1])
prefix = sys.argv[2]
sums = {}
for line in (work / "SHA256SUMS").read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    digest, name = line.split(None, 1)
    sums[name] = digest
for short in ("knolo-infer", "knolo-infer-worker"):
    name = f"{prefix}/{short}"
    if name not in sums:
        raise SystemExit(f"knolo-infer: SHA256SUMS has no entry for {name}")
    got = hashlib.sha256((work / short).read_bytes()).hexdigest()
    if got != sums[name]:
        raise SystemExit(f"knolo-infer: {name} checksum did not match")
PY

bindir="${prefix}/bin"
mkdir -p "${bindir}"
for bin in knolo-infer knolo-infer-worker; do
  stage="${bindir}/.${bin}.installing"
  cp "${work}/${bin}" "${stage}"
  chmod 755 "${stage}"
  mv -f "${stage}" "${bindir}/${bin}"
done

echo "installed ${bindir}/knolo-infer"
echo "installed ${bindir}/knolo-infer-worker"
