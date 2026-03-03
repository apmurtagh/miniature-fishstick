"""
Run ID generation.

IDs are deterministic-ish, lexicographically sortable, and globally unique:

    YYYYMMDD_HHMMSS_<8-char UUID4 hex>

Example: ``20260303_142201_a3f7c1b2``
"""

import uuid
from datetime import datetime, timezone


def generate_run_id(dt: datetime | None = None) -> str:
    """Generate a sortable, unique run identifier.

    Parameters
    ----------
    dt:
        Datetime to embed in the ID.  Defaults to the current UTC time.
        Providing a fixed value makes IDs reproducible in tests.

    Returns
    -------
    str
        A string of the form ``YYYYMMDD_HHMMSS_<8-hex-chars>``.
    """
    if dt is None:
        dt = datetime.now(tz=timezone.utc)
    timestamp_part = dt.strftime("%Y%m%d_%H%M%S")
    unique_part = uuid.uuid4().hex[:8]
    return f"{timestamp_part}_{unique_part}"
