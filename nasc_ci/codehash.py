"""A hash of the compiled code of one commit (v0.3, 24 Nov 2026).

If two commits give the same hash, their compiled classes are byte for byte the same, so a
performance change between them cannot come from the code: any alert on such a commit is a false
alert by construction (e.g. a commit that only edits comments, like the pilot's commit 2).

The hash covers every *.class file under the given folders: its path relative to the folder and
the SHA-256 of its bytes, in sorted order, so it does not depend on file times or listing order.

  python -m nasc_ci codehash jenetics/build/classes/java/main jenetics/build/classes/java/jmh
"""

import hashlib
from pathlib import Path


def tree_hash(root, suffix=".class"):
    """-> {"sha256": hex, "files": number of files} for all files ending in suffix under root."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a folder")
    files = sorted(p for p in root.rglob(f"*{suffix}") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return {"sha256": h.hexdigest(), "files": len(files)}


def code_hash(folders, suffix=".class"):
    """-> {"sha256": combined hex, "parts": {folder name: tree_hash}} for several folders
    (e.g. the library classes and the benchmark classes)."""
    # named by the last three parts of the path (e.g. classes/java/main), so the hash does not
    # depend on where the build folder is, and no local path is stored
    parts = {"/".join(Path(f).resolve().parts[-3:]): tree_hash(f, suffix) for f in folders}
    h = hashlib.sha256()
    for name in sorted(parts):
        h.update(name.encode("utf-8") + b"\0" + parts[name]["sha256"].encode("ascii"))
    return {"sha256": h.hexdigest(), "parts": parts}


def same_code(a, b):
    """True / False if both versions have a code hash, None if one is missing."""
    if not a or not b:
        return None
    return a.get("sha256") == b.get("sha256")
