"""GASST demo: model smoke check or recorded execution of the existing main.py."""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import runpy
import sys
import traceback

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "pu": (1, "PaviaU.mat", "PaviaU_gt.mat"),
    "ip": (4, "Indian_pines_corrected.mat", "Indian_pines_gt.mat"),
    "houston13": (3, "GRSS2013.mat", "GRSS2013_gt.mat"),
    "whuhh": (7, "WHU_Hi_HongHu.mat", "WHU_Hi_HongHu_gt.mat"),
}
SPLIT_SEEDS = [202401, 202402, 202403, 202404, 202405,
               202406, 202407, 10001, 202409, 202410]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Cannot serialize {}".format(type(value).__name__))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=json_default), encoding="utf-8")


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "experiment"], default="smoke")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="pu")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, help="New/empty output directory")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--train-num", type=int, default=30)
    parser.add_argument("--seed", type=int, default=202401,
                        help="Initial Python/NumPy/Torch seed; main.py keeps its split seeds")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the experiment plan without imports, data access or execution")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.train_num < 1 or args.threads < 1:
        parser.error("epochs, train-num and threads must be positive")
    if not 1 <= args.runs <= len(SPLIT_SEEDS):
        parser.error("runs must be between 1 and 10 (main.py's available split seeds)")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    return args


def experiment_argv(args):
    return [str(ROOT / "main.py"), "--model", "GASST",
            "--dataset_name", args.dataset, "--dataID", str(DATASETS[args.dataset][0]),
            "--dataset_dir", str(args.data_dir.resolve()),
            "--epoch", str(args.epochs), "--num_run", str(args.runs),
            "--train_num", str(args.train_num), "--patch_size", "10",
            "--ps", "2", "--pc", "30", "--emb_dim", "128",
            "--depth", "6", "--num_heads", "8", "--group_num", "4",
            "--k", "1/2", "--bs", "256", "--kernel_size", "3",
            "--drop_rate", "0.1", "--save_path_prefix", "./"]


def smoke(torch, output):
    from models.GASST import GASST
    model = GASST(dataset="pu", kernel_size=3, ps=2, k=0.5,
                  group_num=4, emb_dim=128, depth=6, drop_rate=0.1, num_heads=8)
    x = torch.randn(2, 1, 30, 10, 10)
    target = torch.tensor([0, 1])
    model.train()
    logits = model(x)
    assert tuple(logits.shape) == (2, 9), logits.shape
    loss = torch.nn.functional.cross_entropy(logits, target)
    assert torch.isfinite(loss).item(), "Nonfinite loss"
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all().item() for g in gradients)
    model.eval()
    with torch.no_grad():
        prediction = model(x)
    assert torch.isfinite(prediction).all().item()
    result = {"purpose": "Synthetic-input execution check, not classification accuracy",
              "input_shape": list(x.shape), "output_shape": list(logits.shape),
              "loss": loss.item(), "finite_gradients": True,
              "parameters": sum(p.numel() for p in model.parameters())}
    save_json(output / "smoke_result.json", result)
    print(json.dumps(result, indent=2))


def experiment(args, output):
    original_argv, original_cwd = sys.argv[:], Path.cwd()
    try:
        os.chdir(output)
        sys.argv = experiment_argv(args)
        # Execute the source unchanged. Its globals expose the actual reported results.
        result = runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
        report = {"metrics": result["results"], "labels": result["labels"],
                  "options": vars(result["opts"]),
                  "train_seconds": result["train_times"],
                  "test_seconds": result["test_times"],
                  "note": "Original main.py protocol; final in-memory model evaluation"}
        save_json(output / "metrics.json", report)
        import numpy as np
        np.savez_compressed(output / "last_run_split.npz",
                            train_gt=result["train_gt"], test_gt=result["test_gt"])
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)


def main(argv=None):
    args = parse_args(argv)
    plan = {"mode": args.mode, "initial_seed": args.seed,
            "split_seeds": SPLIT_SEEDS[:args.runs] if args.mode == "experiment" else [],
            "device": "cpu", "threads": args.threads,
            "argv": experiment_argv(args) if args.mode == "experiment" else []}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    data_paths = []
    if args.mode == "experiment":
        data_paths = [args.data_dir.resolve() / args.dataset / name
                      for name in DATASETS[args.dataset][1:]]
        missing = [str(path) for path in data_paths if not path.is_file()]
        if missing:
            print("Missing data files:\n" + "\n".join(missing), file=sys.stderr)
            return 2
    output = (args.output or ROOT / "outputs" / (
        "demo_" + args.mode + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        print("Choose a new or empty --output directory: " + str(output), file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(plan, status="running", python=platform.python_version(),
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    files={str(p.relative_to(ROOT)): sha256(p)
                           for folder in [ROOT, ROOT / "models", ROOT / "utils"]
                           for p in sorted(folder.glob("*.py"))},
                    data_sha256={str(p): sha256(p) for p in data_paths})
    save_json(output / "manifest.json", manifest)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("MPLBACKEND", "Agg")
    with (output / "console.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                import numpy as np
                import torch
                manifest["packages"] = {d.metadata["Name"]: d.version
                                        for d in importlib.metadata.distributions()
                                        if d.metadata["Name"]}
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.set_num_threads(args.threads)
                print("Output:", output)
                if args.mode == "smoke":
                    smoke(torch, output)
                else:
                    experiment(args, output)
                manifest["status"] = "completed"
                return 0
            except (Exception, KeyboardInterrupt) as error:
                manifest["status"] = "failed"
                manifest["error"] = "{}: {}".format(type(error).__name__, error)
                traceback.print_exc()
                print("Check console.log and the README environment instructions.")
                return 1
            finally:
                manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
                save_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    sys.exit(main())
