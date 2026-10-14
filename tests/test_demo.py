"""Standard-library tests for demo planning, output protection and failure recording."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gasst_demo", ROOT / "demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class DemoTests(unittest.TestCase):
    def test_dataset_mapping_and_run_bounds(self):
        for name, (data_id, _, _) in demo.DATASETS.items():
            args = demo.parse_args(["--mode", "experiment", "--dataset", name])
            argv = demo.experiment_argv(args)
            self.assertEqual(argv[argv.index("--dataID") + 1], str(data_id))
            self.assertEqual(argv[argv.index("--dataset_name") + 1], name)
            self.assertEqual(argv[argv.index("--pc") + 1], "30")
        for bad in ["0", "11"]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                demo.parse_args(["--runs", bad])

    def test_dry_run_does_not_create_output_or_import_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "unused"
            capture = io.StringIO()
            with mock.patch.dict(sys.modules, {"torch": None}), contextlib.redirect_stdout(capture):
                code = demo.main(["--mode", "experiment", "--dry-run", "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(capture.getvalue())["split_seeds"], [202401])
            self.assertFalse(output.exists())

    def test_missing_data_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result"
            with contextlib.redirect_stderr(io.StringIO()):
                code = demo.main(["--mode", "experiment", "--data-dir", tmp, "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())

    def test_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "keep.txt"
            marker.write_text("existing result", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                code = demo.main(["--output", tmp])
            self.assertEqual(code, 2)
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing result")

    def test_dependency_error_is_recorded_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "failed"
            with mock.patch.dict(sys.modules, {"numpy": None}), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = demo.main(["--output", str(output)])
            self.assertEqual(code, 1)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("ModuleNotFoundError", manifest["error"])
            self.assertIn("Traceback", (output / "console.log").read_text(encoding="utf-8"))

    def test_entrypoint_failure_restores_cwd_and_argv(self):
        before_cwd, before_argv = Path.cwd(), sys.argv[:]
        args = demo.parse_args(["--mode", "experiment"])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)

            def fail(path, run_name):
                self.assertEqual(Path.cwd(), output)
                self.assertEqual(path, str(ROOT / "main.py"))
                self.assertEqual(run_name, "__main__")
                raise RuntimeError("entrypoint failure")

            with mock.patch.object(demo.runpy, "run_path", side_effect=fail), self.assertRaises(RuntimeError):
                demo.experiment(args, output)
        self.assertEqual(Path.cwd(), before_cwd)
        self.assertEqual(sys.argv, before_argv)


if __name__ == "__main__":
    unittest.main()
