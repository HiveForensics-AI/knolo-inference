#!/usr/bin/env bash
# Install the CPU release of knolo-infer and knolo-infer-worker.
# The CUDA archive is published separately and is not selected here.
set -euo pipefail

repo="${KNOLO_INFER_REPO:-HiveForensics-AI/knolo-inference}"
prefix="${KNOLO_INFER_PREFIX:-${HOME}/.local}"
asset="knolo-infer-linux-x86_64.tar.gz"
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
  local base
  if [[ -n "${KNOLO_INFER_DIST_URL:-}" ]]; then
    base="${KNOLO_INFER_DIST_URL%/}"
  elif [[ "${version}" == "latest" ]]; then
    base="https://github.com/${repo}/releases/latest/download"
  else
    base="https://github.com/${repo}/releases/download/${version}"
  fi
  curl --fail --location --silent --show-error --output "${dest}" "${base}/${name}"
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

python3 - "${work}" <<'PY'
import hashlib, pathlib, sys
work = pathlib.Path(sys.argv[1])
sums = {}
for line in (work / "SHA256SUMS").read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    digest, name = line.split(None, 1)
    sums[name] = digest
for name in ("knolo-infer", "knolo-infer-worker"):
    if name not in sums:
        raise SystemExit(f"knolo-infer: SHA256SUMS has no entry for {name}")
    got = hashlib.sha256((work / name).read_bytes()).hexdigest()
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
