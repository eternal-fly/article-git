"""Finish the verified scan after a layout-only export correction."""
from grid_run import *


def main():
    lock = (HERE / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    try:
        freeze()
        assert json.loads((HERE / "numerical_audit.json").read_text())["passed"]
        assert len(pd.read_csv(HERE / "paired_delta_mae.csv")) == 4000
        status("running", "layout-corrected figure export; training already complete", started)
        from grid_plot_v2 import main as plot
        plot()
        status("running", "complete artifact and source audit", started)
        from audit_delivery import main as audit
        audit()
        status("complete", "all training, intervals and exports audited; manual figure review pending", started)
    except BaseException as error:
        status("failed", str(error), started)
        raise


if __name__ == "__main__":
    main()
