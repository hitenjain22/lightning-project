"""Inverse-crime guard: reconstruction code must never touch the ground-truth block."""

import ast
from pathlib import Path

import thunder

RECON_DIR = Path(thunder.__file__).parent / "recon"


def _violations(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and any(a.name == "GroundTruth" for a in node.names):
            found.append(f"line {node.lineno}: imports GroundTruth")
        elif isinstance(node, ast.Attribute) and node.attr == "truth":
            found.append(f"line {node.lineno}: reads .truth")
        elif isinstance(node, ast.Attribute) and node.attr == "GroundTruth":
            found.append(f"line {node.lineno}: uses GroundTruth")
        elif isinstance(node, ast.Name) and node.id == "GroundTruth":
            found.append(f"line {node.lineno}: uses GroundTruth")
    return found


def test_detector_catches_violations():
    assert _violations("from thunder.types import GroundTruth")
    assert _violations("x = recording.truth.t0")
    assert _violations("import thunder.types as T\nT.GroundTruth")
    assert not _violations("t0 = recording.reported_t0")


def test_recon_modules_do_not_access_truth():
    assert RECON_DIR.is_dir()
    problems = {
        str(p.relative_to(RECON_DIR)): v
        for p in RECON_DIR.rglob("*.py")
        if (v := _violations(p.read_text()))
    }
    assert not problems, f"recon/ accesses ground truth: {problems}"
