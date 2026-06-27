from pathlib import Path


def test_expected_project_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = [
        root / "README.md",
        root / "requirements.txt",
        root / "src" / "lane_binary_classifier" / "model.py",
        root / "src" / "lane_binary_classifier" / "train.py",
        root / "src" / "lane_binary_classifier" / "evaluate.py",
        root / "src" / "lane_binary_classifier" / "predict_onnx.py",
        root / "scripts" / "make_toy_dataset.py",
        root / "scripts" / "prepare_tusimple_binary.py",
    ]
    missing = [path for path in expected if not path.exists()]
    assert not missing


if __name__ == "__main__":
    test_expected_project_files_exist()
