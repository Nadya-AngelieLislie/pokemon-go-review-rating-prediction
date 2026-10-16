import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from experiment_tracking import track_run


def test_track_run_works_with_or_without_mlflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with track_run("smoke", params={"a": 1}) as log:
        log(metrics={"score": 1.0})
