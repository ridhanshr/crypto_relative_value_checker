"""Crash-safe artifact writes for Quantara consumption.

Quantara reads checker JSON straight off disk. A crash mid-write must NEVER
leave a half-written file behind: write to a temp file in the SAME
directory, flush + fsync to durable storage, then os.replace() (atomic on
both POSIX and Windows/NTFS). Readers always see the old valid file or the
new valid file -- never a torn one.
"""

import json
import os
import tempfile
from pathlib import Path


def atomic_write_text(path, text, encoding="utf-8"):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(target)


def atomic_write_json(path, obj):
    return atomic_write_text(path, json.dumps(obj, indent=2, default=str))
