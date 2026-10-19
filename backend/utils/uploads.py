"""Shared upload utilities for WIREFRAME."""

import os
import re
import uuid
from pathlib import Path

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


def validate_file_extension(ext: str) -> str:
    """Validate and normalize a file extension against allowed image types."""
    if not ext:
        raise ValueError("Filename must have an extension")
    normalized = ext.lower()
    if normalized not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Disallowed file extension: '{normalized}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
    return normalized


def generate_safe_filename(name: str) -> str:
    """Generate a collision-free, path-traversal-safe, sanitized filename.

    Format: {8 hex of uuid4}_{sanitized stem}{extension}

    Handles:
    - Empty or whitespace-only names (raises ValueError)
    - Names consisting only of dots without extension (raises ValueError)
    - Missing extension (raises ValueError)
    - Disallowed extensions (raises ValueError)
    - Windows reserved names (CON, NUL, PRN, etc. - prefixed with upload_)
    - Names with stems over 100 characters (truncated to 100)
    - Unicode and special characters (sanitized to safe ASCII alphanumeric, dots, hyphens, underscores)
    - Empty or dot-only stem with valid extension (falls back stem to 'upload')
    """
    if not name or not str(name).strip():
        raise ValueError("Filename cannot be empty")

    # Strip directory components to prevent path traversal
    base = os.path.basename(str(name).strip())

    # If the base consists only of dots (e.g. '.', '..', '...')
    if re.fullmatch(r'\.+', base):
        raise ValueError("Filename cannot consist only of dots")

    # Split stem and extension
    if '.' not in base:
        raise ValueError("Filename must have an extension")

    stem, _, ext_part = base.rpartition('.')
    if not ext_part:
        raise ValueError("Filename must have an extension")

    ext = f".{ext_part}"

    # Validate extension against allowed set
    norm_ext = validate_file_extension(ext)

    # Sanitize stem: replace any character not in [a-zA-Z0-9_.-] with '_'
    sanitized_stem = re.sub(r'[^a-zA-Z0-9_.-]', '_', stem)

    # If stem is empty, only dots, only underscores, or blank, fallback to 'upload'
    if not sanitized_stem or not sanitized_stem.strip("._"):
        sanitized_stem = "upload"

    # Truncate stem to maximum 100 characters
    if len(sanitized_stem) > 100:
        sanitized_stem = sanitized_stem[:100]

    # Handle Windows reserved device names
    if sanitized_stem.upper() in WINDOWS_RESERVED_NAMES:
        sanitized_stem = f"upload_{sanitized_stem}"

    hex8 = uuid.uuid4().hex[:8]
    return f"{hex8}_{sanitized_stem}{norm_ext}"


def is_safe_contained_path(path_str: str | Path | None, allowed_dirs: tuple[str, ...] = ("uploads", "outputs")) -> bool:
    """Return True if path_str resolves strictly inside one of allowed_dirs."""
    if not path_str:
        return False
    try:
        p = Path(path_str).resolve()
        for base_name in allowed_dirs:
            base = Path(base_name).resolve()
            try:
                if p.is_relative_to(base) and p != base:
                    return True
            except AttributeError:
                try:
                    p.relative_to(base)
                    if p != base:
                        return True
                except ValueError:
                    continue
    except Exception:
        return False
    return False

