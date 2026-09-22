import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "eval/scripts/cluster3_efficiency_preflight.py"
SPEC = importlib.util.spec_from_file_location("cluster3_efficiency_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_checkpoint(root: Path, *, architecture=None, dtype="BF16") -> Path:
    root.mkdir()
    config = {
        "architectures": [architecture or MODULE.EXPECTED_ARCHITECTURE],
        **MODULE.QWEN35_2B_FINGERPRINT,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text("{}", encoding="utf-8")
    (root / "chat_template.jinja").write_text("template", encoding="utf-8")
    header = json.dumps(
        {"weight": {"dtype": dtype, "shape": [1], "data_offsets": [0, 2]}},
        separators=(",", ":"),
    ).encode("utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\x00\x00"
    )
    return root


def test_checkpoint_inventory_architecture_scale_and_bf16(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "model")
    identity = MODULE.checkpoint_identity(checkpoint)
    assert identity["architecture"] == MODULE.EXPECTED_ARCHITECTURE
    assert identity["model_scale"] == "2B"
    assert identity["weights"] == {
        "tensor_count": 1,
        "dtype_counts": {"BF16": 1},
    }
    assert set(identity["files"]) == set(MODULE.REQUIRED_CHECKPOINT_FILES)
    assert all(len(value["sha256"]) == 64 for value in identity["files"].values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "missing required files"),
        ("architecture", "architecture mismatch"),
        ("scale", "scale mismatch"),
        ("dtype", "not uniformly BF16"),
    ),
)
def test_checkpoint_validation_fails_closed(tmp_path, mutation, message):
    checkpoint = write_checkpoint(
        tmp_path / "model",
        architecture=("WrongArchitecture" if mutation == "architecture" else None),
        dtype=("F16" if mutation == "dtype" else "BF16"),
    )
    if mutation == "missing":
        (checkpoint / "chat_template.jinja").unlink()
    elif mutation == "scale":
        config_path = checkpoint / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["hidden_size"] = 4096
        config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        MODULE.checkpoint_identity(checkpoint)


def test_frozen_provenance_requires_frozen_sha_and_every_hash(tmp_path):
    hashes = {
        name: str(index) * 64
        for index, name in enumerate(MODULE.REQUIRED_CHECKPOINT_FILES, 1)
    }
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "revision": MODULE.FROZEN_SHA,
                "checkpoint": {
                    name: {"sha256": value} for name, value in hashes.items()
                },
            }
        ),
        encoding="utf-8",
    )
    assert MODULE.load_frozen_provenance([evidence]) == hashes

    evidence.write_text(
        json.dumps({"revision": "wrong", "checkpoint": hashes}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="no supplied provenance"):
        MODULE.load_frozen_provenance([evidence])


def test_conflicting_frozen_hashes_are_rejected(tmp_path):
    paths = []
    for index, digest in enumerate(("a" * 64, "b" * 64)):
        path = tmp_path / f"evidence-{index}.json"
        inventory = {
            name: {"sha256": "c" * 64} for name in MODULE.REQUIRED_CHECKPOINT_FILES
        }
        inventory["config.json"] = {"sha256": digest}
        path.write_text(
            json.dumps({"revision": MODULE.FROZEN_SHA, "checkpoint": inventory}),
            encoding="utf-8",
        )
        paths.append(path)
    with pytest.raises(RuntimeError, match="conflicting checkpoint hashes"):
        MODULE.load_frozen_provenance(paths)
