"""Probe the current digest implementations without importing GPU entry points.

This executes selected, unchanged AST definitions. It does not run the exporter,
S0 orchestration, model inference, or the repository test suite.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[3]
BRIDGE = REPO / "earthdelta/bridge/stormer_bridge.py"
EXPORTER = REPO / "scripts/export_upstream_reference.py"


def load_selected(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if node.name in names:
                selected.append(node)
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
                selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def compare(norm_dir, variables, intervals, namespace):
    norm = namespace["NormalizationContract"].from_npz_dir(
        str(norm_dir), variables=variables, intervals=intervals,
        policy=namespace["POLICY_OFFICIAL_ZERO_DIFF_MEAN"],
    )
    exported = namespace["compute_normalization_digest"](norm_dir, variables, intervals)
    return {
        "intervals": list(intervals),
        "variables_count": len(variables),
        "bridge_digest": norm.digest,
        "exporter_digest": exported,
        "equal": norm.digest == exported,
    }


def main():
    namespace = dict(globals())
    load_selected(BRIDGE, {
        "NormalizationContract", "DEFAULT_VARIABLES", "CONSTANTS",
        "POLICY_LEGACY", "POLICY_OFFICIAL_ZERO_DIFF_MEAN",
    }, namespace)
    load_selected(EXPORTER, {"compute_normalization_digest"}, namespace)

    cases = []
    for dtype in (np.float32, np.float64):
        with tempfile.TemporaryDirectory(prefix="earthdelta-digest-probe-") as temp:
            norm_dir = Path(temp)
            variables = ["a", "b"]

            def save(name, values):
                np.savez(norm_dir / name, **{
                    var: np.array([value], dtype=dtype)
                    for var, value in zip(variables, values)
                })

            save("normalize_mean.npz", (1.0, 2.0))
            save("normalize_std.npz", (2.0, 4.0))
            for interval in (6, 24):
                save(f"normalize_diff_mean_{interval}.npz", (0.1, 0.2))
                save(f"normalize_diff_std_{interval}.npz", (0.5, 0.75))
            for intervals in ((6,), (6, 24)):
                cases.append({
                    "case": "synthetic", "npz_dtype": np.dtype(dtype).name,
                    **compare(norm_dir, variables, intervals, namespace),
                })

    asset_dir = REPO / "reference/stormer/normalization_constants"
    cases.append({
        "case": "repository_npz", "path": str(asset_dir.relative_to(REPO)),
        **compare(asset_dir, namespace["DEFAULT_VARIABLES"], (6, 24), namespace),
    })
    result = {
        "head": subprocess.check_output(
            ["git", "--git-dir=.git", "--work-tree=.", "rev-parse", "HEAD"],
            cwd=REPO, text=True,
        ).strip(),
        "method": "AST extraction of unchanged normalization class and digest function",
        "limits": "No production imports, GPU inference, S0 orchestration, or pytest run",
        "numpy": np.__version__, "torch": torch.__version__,
        "source_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (BRIDGE, EXPORTER)
        },
        "cases": cases,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
