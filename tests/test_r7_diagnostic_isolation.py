from __future__ import annotations

import json

import pytest

from scripts import r5_diagnose


def _inputs(tmp_path):
    paths = {}
    for key in ("protocol", "cache_manifest", "prediction_freeze", "evaluation"):
        p = tmp_path / f"{key}.json"
        p.write_text(json.dumps({"key": key}))
        paths[key] = str(p)
    x = tmp_path / "inputs.json"
    x.write_text(json.dumps(paths))
    return x, paths


def test_diagnostic_protocol_rejects_swapped_input_path(tmp_path):
    inputs, paths = _inputs(tmp_path)
    proto_out = tmp_path / "proto"
    r5_diagnose.freeze(type("Args", (), {"inputs": inputs, "out": proto_out})())
    swapped = tmp_path / "swapped.json"
    swapped.write_text(json.dumps({**paths, "evaluation": paths["protocol"]}))
    with pytest.raises(ValueError, match="not the frozen path"):
        r5_diagnose.diagnose(type("Args", (), {
            "inputs": swapped,
            "protocol": proto_out / "protocol.json",
            "out": tmp_path / "result",
        })())


def test_diagnostic_protocol_rejects_changed_frozen_child(tmp_path):
    inputs, paths = _inputs(tmp_path)
    proto_out = tmp_path / "proto"
    r5_diagnose.freeze(type("Args", (), {"inputs": inputs, "out": proto_out})())
    Path = __import__("pathlib").Path
    Path(paths["protocol"]).write_text(json.dumps({"changed": True}))
    with pytest.raises(ValueError, match="input changed"):
        r5_diagnose.diagnose(type("Args", (), {
            "inputs": inputs,
            "protocol": proto_out / "protocol.json",
            "out": tmp_path / "result",
        })())
