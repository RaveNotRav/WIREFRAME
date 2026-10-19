import re
import sqlite3
from typing import Any

from fastapi import HTTPException


def validate_gps_coordinates(lat_raw: Any, lng_raw: Any) -> tuple[float | None, float | None, str | None]:
    """Validate GPS latitude and longitude coordinates.

    - Both must be present or both absent (else 422)
    - Numeric parse errors -> 422
    - Range checks: latitude within [-90, 90], longitude within [-180, 180] (else 422)
    - Outside Philippine bounds (4.5–21.5 / 114.0–127.0) returns a warning string
    Returns: (parsed_lat, parsed_lng, warning)
    """
    def parse_coord(val: Any, name: str) -> float | None:
        if val is None:
            return None
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return None
        try:
            return float(val)
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail=f"Invalid GPS {name} (must be float)")

    parsed_lat = parse_coord(lat_raw, "latitude")
    parsed_lng = parse_coord(lng_raw, "longitude")

    if (parsed_lat is None) != (parsed_lng is None):
        raise HTTPException(status_code=422, detail="Both latitude and longitude must be provided, or neither")

    if parsed_lat is not None and not (-90.0 <= parsed_lat <= 90.0):
        raise HTTPException(status_code=422, detail="Invalid latitude")

    if parsed_lng is not None and not (-180.0 <= parsed_lng <= 180.0):
        raise HTTPException(status_code=422, detail="Invalid longitude")

    warning = None
    if parsed_lat is not None and parsed_lng is not None:
        if not (4.5 <= parsed_lat <= 21.5) or not (114.0 <= parsed_lng <= 127.0):
            warning = "GPS coordinates are outside the Philippines."

    return parsed_lat, parsed_lng, warning



def validate_location(company: str | None, barangay: str | None, street: str | None) -> tuple[str, str, str]:
    """Validates and cleans location data from capture form."""
    def clean(s: str | None) -> str:
        if not s:
            return ""
        # Remove control characters
        s = re.sub(r'[\x00-\x1F\x7F]', '', s)
        # Collapse whitespace
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    comp = clean(company)
    brgy = clean(barangay)
    st = clean(street)

    errors = []
    
    if not comp:
        errors.append("Company is required")
    elif len(comp) < 2 or len(comp) > 100:
        errors.append("Company must be between 2 and 100 characters")
        
    if not brgy:
        errors.append("Barangay is required")
    elif len(brgy) < 2 or len(brgy) > 80:
        errors.append("Barangay must be between 2 and 80 characters")
        
    if not st:
        errors.append("Street is required")
    elif len(st) < 2 or len(st) > 120:
        errors.append("Street must be between 2 and 120 characters")
        
    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))
        
    return comp, brgy, st


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    # Strip prefix. Use non-word boundary \W or just plain prefix
    s = re.sub(r'^(brgy\.?|barangay)\s*', '', s).strip()
    return s


def get_or_create_site(conn: sqlite3.Connection, company: str, barangay: str, street: str, gps_lat: float | None = None, gps_lng: float | None = None) -> int:
    """Get or create site by normalized barangay/street."""
    brgy_norm = _normalize(barangay)
    st_norm = _normalize(street)

    # We use INSERT OR IGNORE, which relies on the unique index on barangay_norm, street_norm.
    conn.execute(
        """INSERT OR IGNORE INTO sites (name, barangay, street, barangay_norm, street_norm) 
           VALUES (?, ?, ?, ?, ?)""",
        (f"{barangay} - {street}", barangay, street, brgy_norm, st_norm)
    )
    
    # SELECT to get the ID
    row = conn.execute(
        "SELECT id, lat, lng FROM sites WHERE barangay_norm = ? AND street_norm = ?",
        (brgy_norm, st_norm)
    ).fetchone()
    
    site_id = row["id"]
    
    # Update GPS if missing
    if gps_lat is not None and gps_lng is not None and (row["lat"] is None or row["lng"] is None):
        conn.execute(
            "UPDATE sites SET lat = ?, lng = ? WHERE id = ?",
            (gps_lat, gps_lng, site_id)
        )
        conn.commit()
        
    return site_id
