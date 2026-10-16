"""Optional experiment tracking.

Wraps MLflow when it is installed and silently does nothing otherwise, so
notebooks and scripts run the same on any machine.

    from experiment_tracking import track_run
    with track_run("baseline", params={"lr": 1e-3}) as log:
        ...  # train
        log(metrics={"val_f1": 0.91})
"""
from contextlib import contextmanager

try:
    import mlflow
except ImportError:  # MLflow is optional
    mlflow = None


@contextmanager
def track_run(run_name, params=None, experiment="portfolio"):
    if mlflow is None:
        yield lambda metrics=None, artifacts=None: None
        return
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)

        def log(metrics=None, artifacts=None):
            if metrics:
                mlflow.log_metrics(metrics)
            for path in artifacts or []:
                mlflow.log_artifact(path)

        yield log
