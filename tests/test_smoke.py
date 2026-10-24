"""Smoke tests — no dataset or trained model required.

Run:  pytest tests/ -v
TensorFlow-dependent tests are skipped automatically when TF is absent.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_config_loads():
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["backbone"] in {"efficientnetb0", "mobilenetv2"}
    assert len(cfg["data"]["image_size"]) == 2
    assert cfg["training"]["epochs"] > 0


def test_data_utilities_on_synthetic_folders(tmp_path):
    tf = pytest.importorskip("tensorflow")
    assert tf is not None
    from PIL import Image

    from dental_scan.data import count_images, list_classes

    for cls in ("caries", "healthy"):
        folder = tmp_path / cls
        folder.mkdir()
        Image.new("RGB", (32, 32), color="gray").save(folder / "sample.jpg")

    assert list_classes(tmp_path) == ["caries", "healthy"]
    assert count_images(tmp_path) == {"caries": 1, "healthy": 1}


def test_model_builder_creates_model_with_tiny_input():
    pytest.importorskip("tensorflow")
    from dental_scan.model import build_model

    model = build_model(num_classes=4, image_size=(64, 64), backbone="mobilenetv2")
    assert model.output_shape == (None, 4)
    assert model.input_shape == (None, 64, 64, 3)


def test_model_builder_rejects_bad_inputs():
    pytest.importorskip("tensorflow")
    from dental_scan.model import build_model

    with pytest.raises(ValueError):
        build_model(num_classes=1)
    with pytest.raises(ValueError):
        build_model(num_classes=4, backbone="not-a-backbone")


def test_predict_missing_model_raises_clear_error(tmp_path):
    pytest.importorskip("tensorflow")
    from dental_scan.predict import predict_image

    with pytest.raises(FileNotFoundError):
        predict_image(str(tmp_path / "nope.keras"), str(tmp_path / "x.jpg"))
