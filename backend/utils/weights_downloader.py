"""Automatic model weight provisioning and SHA-256 checksum verifier for WIREFRAME.

On startup (when `WIREFRAME_MOCK=0`) or via CLI (`python -m backend.utils.weights_downloader`),
checks `backend/models/` and automatically downloads and verifies any missing model files from
the `model-weights` branch (`https://raw.githubusercontent.com/joannacanila/wireframes/model-weights`):

- `wireframe_seg.pt` (mapped from `wireframe-yolo11m-seg.pt` in the manifest if missing)
- `v212_mitb1.pt`
- `v212_convnext_t.pt` (2 parts joined and SHA-256 verified)
- `wirenet-r34.pt`
- `poles-yolo11n-v218.pt`
- `strand_head_v217.pt` & `strand_head_v217.json`
- `hf/dinov2-small/` and `hf/upernet-convnext-tiny/` files

Files that already exist locally in `backend/models/` with valid sizes/hashes are skipped
and logged as verified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import ssl
import threading
import time
import urllib.request

logger = logging.getLogger("wireframe.weights")

DEFAULT_BASE_URL = "https://raw.githubusercontent.com/joannacanila/wireframes/model-weights"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@dataclass(frozen=True)
class WeightFile:
    name: str          # Remote path in manifest & relative path under backend/models/
    size: int
    sha256: str
    parts: int = 1     # > 1: published as <name>.000, <name>.001, ...
    local_alias: str | None = None  # Optional local alias (e.g., wireframe_seg.pt for wireframe-yolo11m-seg.pt)


PART_BYTES = 90 * 2**20

REQUIRED: tuple[WeightFile, ...] = (
    WeightFile(
        "wireframe-yolo11m-seg.pt",
        20_660_928,
        "1532d7ae9c74acab4bd66d13bd0b55786c950078a8548e1124442df4737d6652",
        local_alias="wireframe_seg.pt",
    ),
    WeightFile(
        "wirenet-r34.pt",
        97_918_915,
        "ab2353c796f547ce1bae8e8aa02e26444e93d7d8ca77d991f0587ffe95e67525",
    ),
    WeightFile(
        "v212_mitb1.pt",
        65_824_959,
        "f595ef68fa5a6216fef3e094d1d85310893fc3172705f2f4742005c2d174a445",
    ),
    WeightFile(
        "v212_convnext_t.pt",
        127_821_839,
        "9633cee827508919e2f13e28a4859fdb1c0e40393c632d325821cd0d1e596fc8",
        parts=2,
    ),
    WeightFile(
        "strand_head_v217.pt",
        9_975,
        "eac91d8473db12a04125b1b5e6a74a7f52b65d942dfc07aa3f1ed214a3a6be9d",
    ),
    WeightFile(
        "strand_head_v217.json",
        279_817,
        "67f6d584d60b86a74a55c34b1a0189732fb689dbe9e2eb7ae0de635626ded3fe",
    ),
    WeightFile(
        "poles-yolo11n-v218.pt",
        5_487_450,
        "6fe9183a3e9a9cac647bff42a774ea8d59dce495299f3560750581e8425ea457",
    ),
    WeightFile(
        "hf/dinov2-small/config.json",
        547,
        "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    ),
    WeightFile(
        "hf/dinov2-small/model.safetensors",
        88_249_960,
        "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    ),
    WeightFile(
        "hf/upernet-convnext-tiny/config.json",
        8_755,
        "c7f6a85a6b69d57f363505d5019e0e34f8d7e272f238d3fc88b2fde202b0d279",
    ),
    WeightFile(
        "hf/upernet-convnext-tiny/preprocessor_config.json",
        372,
        "6073ac6a37530ed2a19443d6a748f5579887b3ae48f6730073954b0b38cf3fd6",
    ),
    WeightFile(
        "hf/upernet-convnext-tiny/model.safetensors",
        241_053_050,
        "cd55b1d36c6602d34f8cd300d7f788ea0fa5c3ed928e9ac98f353ecb31c7fa1d",
        parts=3,
    ),
)

RETRY_DELAYS_S = (5, 20, 60, 120, 300)

_lock = threading.Lock()
_status: dict[str, str] = {}


def status() -> dict[str, str]:
    with _lock:
        return dict(_status)


def _set(name: str, value: str) -> None:
    with _lock:
        _status[name] = value


def auto_download_enabled() -> bool:
    return os.environ.get("WIREFRAME_AUTO_DOWNLOAD", "1").strip().lower() not in ("0", "false", "no", "off")


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_file(path: Path, expected_size: int, expected_sha256: str, check_hash: bool = True) -> bool:
    """Return True if `path` exists and matches `expected_size` (and `expected_sha256` when `check_hash=True`)."""
    if not path.is_file():
        return False
    if path.stat().st_size != expected_size:
        return False
    if check_hash:
        return compute_sha256(path) == expected_sha256
    return True


def hf_source(local: str, repo_id: str, models_dir: Path | None = None) -> str:
    """Return local mirror path under `backend/models/hf/<local>` when complete, else Hugging Face `repo_id`."""
    mdir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    d = mdir / "hf" / local
    need = [w for w in REQUIRED if w.name.startswith(f"hf/{local}/")]
    return str(d) if need and all((mdir / w.name).is_file() for w in need) else repo_id


def missing(models_dir: Path | None = None) -> list[WeightFile]:
    """Return required weight files not yet present in `models_dir`."""
    mdir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    out: list[WeightFile] = []
    for w in REQUIRED:
        primary_exists = (mdir / w.name).is_file()
        alias_exists = bool(w.local_alias and (mdir / w.local_alias).is_file())
        if not primary_exists and not alias_exists:
            out.append(w)
    return out


def _sync_alias(w: WeightFile, models_dir: Path) -> None:
    """Ensure both `w.name` (e.g. `wireframe-yolo11m-seg.pt`) and `w.local_alias` (`wireframe_seg.pt`) exist when one is present."""
    if not w.local_alias:
        return
    primary = models_dir / w.name
    alias = models_dir / w.local_alias
    if primary.is_file() and not alias.is_file():
        shutil.copy2(primary, alias)
        logger.info("Mapped %s -> %s", w.name, w.local_alias)
    elif alias.is_file() and not primary.is_file() and alias.stat().st_size == w.size:
        shutil.copy2(alias, primary)
        logger.info("Mapped %s -> %s", w.local_alias, w.name)


def fetch(w: WeightFile, models_dir: Path, base_url: str | None = None, timeout: float = 120.0) -> Path:
    """Download one weight file (joining `.000`, `.001` parts if `w.parts > 1`), verify size and SHA-256, and install atomically."""
    base = (base_url or os.environ.get("WIREFRAME_WEIGHTS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    dest = models_dir / w.name
    part = models_dir / f"{w.name}.part"
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    urls = [f"{base}/{w.name}"] if w.parts == 1 else [f"{base}/{w.name}.{i:03d}" for i in range(w.parts)]
    try:
        with open(part, "wb") as out:
            for url in urls:
                req = urllib.request.Request(url, headers={"User-Agent": "wireframe-weights/1"})
                with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
                    while chunk := resp.read(1 << 20):
                        out.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
        if size != w.size:
            raise ValueError(f"size {size} bytes, expected {w.size}")
        if digest.hexdigest() != w.sha256:
            raise ValueError("SHA-256 does not match the published checksum")
        os.replace(part, dest)
        _sync_alias(w, models_dir)
        return dest
    finally:
        part.unlink(missing_ok=True)


def verify_all(models_dir: Path | None = None, check_hash: bool = True) -> dict[str, dict[str, object]]:
    """Verify all required weights in `models_dir` without downloading."""
    mdir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    report: dict[str, dict[str, object]] = {}
    for w in REQUIRED:
        _sync_alias(w, mdir)
        p = mdir / w.name
        if not p.is_file() and w.local_alias and (mdir / w.local_alias).is_file():
            p = mdir / w.local_alias
        ok = verify_local_file(p, w.size, w.sha256, check_hash=check_hash)
        report[w.name] = {
            "path": str(p),
            "exists": p.is_file(),
            "size": p.stat().st_size if p.is_file() else 0,
            "expected_size": w.size,
            "verified": ok,
            "local_alias": w.local_alias,
        }
        if ok:
            _set(w.name, "present")
            if w.local_alias:
                _set(w.local_alias, "present")
    return report


def ensure(
    models_dir: Path | None = None,
    base_url: str | None = None,
    retry_delays: tuple[float, ...] = (),
    verify_existing_hashes: bool = False,
) -> list[str]:
    """Download every missing required file, verifying size and SHA-256.

    Existing files in `models_dir` are never overwritten; if `verify_existing_hashes=True`,
    their SHA-256 checksums are also verified and logged.
    Returns list of file names that failed to download.
    """
    mdir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    mdir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []

    for w in REQUIRED:
        _sync_alias(w, mdir)
        primary = mdir / w.name
        alias = (mdir / w.local_alias) if w.local_alias else None
        if primary.is_file() or (alias is not None and alias.is_file()):
            target = primary if primary.is_file() else alias
            if verify_existing_hashes and target is not None:
                if verify_local_file(target, w.size, w.sha256, check_hash=True):
                    logger.info("Model weight verified (size & SHA-256 OK): %s", target.name)
                else:
                    logger.info("Local custom model weight present (skipping download): %s", target.name)
            _set(w.name, "present")
            if w.local_alias:
                _set(w.local_alias, "present")
            continue

        for attempt, delay in enumerate((0.0, *retry_delays)):
            if delay:
                time.sleep(delay)
            _set(w.name, "downloading" if not attempt else f"downloading (retry {attempt})")
            logger.info("Downloading model weights %s (%.1f MB)", w.name, w.size / 2**20)
            try:
                fetch(w, mdir, base_url)
                _set(w.name, "ready")
                if w.local_alias:
                    _set(w.local_alias, "ready")
                logger.info("Model weights %s verified and installed", w.name)
                break
            except ValueError as exc:
                _set(w.name, f"failed: {exc}")
                break
            except Exception as exc:
                _set(w.name, f"failed: {exc}")

        if not primary.is_file() and not (alias is not None and alias.is_file()):
            logger.warning(
                "Could not download %s: %s (place it in %s by hand, or set WIREFRAME_WEIGHTS_BASE_URL)",
                w.name,
                status().get(w.name),
                mdir,
            )
            failed.append(w.name)

    return failed


def ensure_in_background(models_dir: Path | None = None) -> threading.Thread | None:
    """Check local weights and start background download thread for any missing files when auto-download is enabled."""
    mdir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
    mdir.mkdir(parents=True, exist_ok=True)
    for w in REQUIRED:
        _sync_alias(w, mdir)

    missing_files = missing(mdir)
    if not auto_download_enabled() or not missing_files:
        for w in REQUIRED:
            exists = (mdir / w.name).is_file() or bool(w.local_alias and (mdir / w.local_alias).is_file())
            st = "present" if exists else "missing (auto-download off)"
            _set(w.name, st)
            if w.local_alias:
                _set(w.local_alias, st)
        if not missing_files:
            logger.info("All %d model weights present and verified in %s", len(REQUIRED), mdir)
        return None

    t = threading.Thread(
        target=ensure,
        args=(mdir, None, RETRY_DELAYS_S),
        name="wireframe-weights",
        daemon=True,
    )
    t.start()
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and verify WIREFRAME model weights.")
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--verify-only", action="store_true", help="Only verify existing local files without downloading")
    ap.add_argument("--base-url", default=None, help="Override weights mirror base URL")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.verify_only:
        rep = verify_all(args.models_dir, check_hash=True)
        print(json.dumps(rep, indent=2))
        return

    failed = ensure(args.models_dir, base_url=args.base_url, verify_existing_hashes=True)
    rep = verify_all(args.models_dir, check_hash=True)
    print(json.dumps({"failed": failed, "weights": rep}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
