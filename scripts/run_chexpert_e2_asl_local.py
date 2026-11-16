"""Download CheXpert, execute the ASL notebook, and save the BCE comparison."""

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/train_chexpert_e2_asl_compare_bce.ipynb"


def main():
    os.chdir(ROOT)
    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.is_relative_to(ROOT / "outputs"):
        raise ValueError("Run directory must be inside the project's outputs directory.")
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    status = {"pid": os.getpid(), "notebook": str(NOTEBOOK), "run_dir": str(run_dir)}

    def save_status(phase, **fields):
        status.update(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **fields)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(status_path)

    try:
        os.environ["KAGGLEHUB_CACHE"] = str(ROOT / "data/.cache/kagglehub")
        os.environ["TORCH_HOME"] = str(ROOT / "data/.cache/torch")
        os.environ["CHEXPERT_OUTPUT_ROOT"] = str(run_dir)
        save_status("downloading", dataset="ashery/chexpert")
        import kagglehub

        print("Downloading ashery/chexpert", flush=True)
        download_path = Path(kagglehub.dataset_download("ashery/chexpert"))
        candidates = list(download_path.glob("train.csv")) + list(download_path.glob("*/train.csv"))
        if not candidates:
            candidates = list(download_path.rglob("train.csv"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one train.csv under {download_path}; found {candidates}")
        data_root = candidates[0].parent
        os.environ["CHEXPERT_DATA_ROOT"] = str(data_root)
        print("DATA_ROOT:", data_root, flush=True)

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for this local training run.")
        save_status("preparing", dataset_path=str(download_path), data_root=str(data_root), gpu=torch.cuda.get_device_name(0))
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        namespace = {"__name__": "__main__"}
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            phase = "training" if source.startswith("# 15. Train") else "evaluating" if index > 24 else "preparing"
            save_status(phase, cell=index, artifacts_dir=str(namespace.get("OUT_DIR", "")))
            print(f"\n[{datetime.now(timezone.utc).isoformat()}] Cell {index}: {source.splitlines()[0]}", flush=True)
            exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)
        save_status("complete", artifacts_dir=str(namespace["OUT_DIR"]), report=str(namespace["OUT_DIR"] / "comparison_report.md"))
        print("Experiment complete:", status["report"], flush=True)
    except BaseException as exc:
        save_status("failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
