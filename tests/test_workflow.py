"""离线迁移、同条件对比及批量流程的失败保护。"""

import json
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from report import build_report, compare_results, load_result
from sample4geo.evaluate.metrics import score_query, summarize
from scripts import run_experiments


def result_fixture(directory, source, correct):
    directory.mkdir(parents=True, exist_ok=True)
    gallery = []
    for label in ("a", "b"):
        image = source / label / "image.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "red").save(image)
        gallery.append({"path": str(image), "key": f"{label}/image.png", "id": label, "included": True})
    queries = []
    for i, record in enumerate(gallery):
        scores = [0.8, 0.8]
        scores[i if correct[i] else 1 - i] = 0.9
        detail = score_query(scores, ["a", "b"], record["id"], 2)
        candidate = lambda idx: {**gallery[idx], "score": scores[idx], "correct": idx == i}
        detail["top_k"] = [candidate(idx) for idx in detail.pop("top_indices")]
        detail["best_positive"] = candidate(detail.pop("best_positive_index"))
        detail["best_negative"] = candidate(detail.pop("best_negative_index"))
        queries.append({**record, **detail})
    data = {"queries": queries, "gallery": gallery,
            "evaluation": {"dataset": "U1652", "direction": "D2S", "altitude": None, "img_size": 384, "data_root": str(source)}}
    metrics = {**summarize(queries, 2), "query_count": 2, "gallery_count": 2, "protocol": {"gallery_scope": "query_ids"}}
    (directory / "queries.json").write_text(json.dumps(data))
    (directory / "metrics.json").write_text(json.dumps(metrics))
    (directory / "train.csv").write_text("epoch,loss,steps,optimizer_updates,amp_skipped_steps\n1,0.5,2,2,0\n")


class WorkflowTests(unittest.TestCase):
    def test_transferred_reports_compare_without_source_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, correct in (("before", (False, True)), ("after", (True, False))):
                directory = root / "server" / name
                result_fixture(directory, root / "dataset", correct)
                build_report(directory)
                with zipfile.ZipFile(directory / "report.zip") as archive:
                    archive.extractall(root / "local" / name)
                    self.assertIn("queries.json", archive.namelist())
                    self.assertIn("train.csv", archive.namelist())
                    self.assertFalse(any(p.endswith(".pth") for p in archive.namelist()))
            shutil.rmtree(root / "server")
            shutil.rmtree(root / "dataset")
            report = build_report(root / "local/after", compare=root / "local/before")
            comparison = json.loads((report / "comparison.json").read_text())
            self.assertEqual(comparison["counts"]["fixed"], 1)
            self.assertEqual(comparison["counts"]["regressed"], 1)
            self.assertEqual(comparison["delta"]["recall@1"], 0)
            self.assertEqual(len(list((report / "images").glob("*.jpg"))), 2)

    def test_comparison_rejects_changed_gallery_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_fixture(root / "before", root / "dataset", (True, True))
            result_fixture(root / "after", root / "dataset", (False, True))
            before, bm = load_result(root / "before")
            after, am = load_result(root / "after")
            bm["protocol"]["gallery_scope"] = "all"
            with self.assertRaisesRegex(ValueError, "协议"):
                compare_results(after, am, before, bm)
            bm["protocol"] = dict(am["protocol"])
            before["gallery"][0]["included"] = False
            with self.assertRaisesRegex(ValueError, "图库"):
                compare_results(after, am, before, bm)

    def test_matrix_packages_both_directions_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_run(command, **kwargs):
                if command[1] == "train.py":
                    (root / "outputs/matrix_u1652").mkdir()
                    (root / "outputs/matrix_u1652/model.pth").write_bytes(b"placeholder")
                elif command[1] == "evaluate.py":
                    output = Path(command[command.index("--output") + 1])
                    result_fixture(output, root / "dataset", (True, False))
                elif command[1] == "report.py":
                    build_report(command[command.index("--experiment") + 1])
                return subprocess.CompletedProcess(command, 0)

            with patch.object(run_experiments, "ROOT", root), patch("sys.argv", ["runner", "--name", "matrix", "--datasets", "U1652"]), patch.object(run_experiments.subprocess, "run", side_effect=fake_run):
                run_experiments.main()
            status = json.loads((root / "outputs/matrix/run.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertEqual(len(status["results"]), 2)
            with zipfile.ZipFile(root / "outputs/matrix/results.zip") as archive:
                self.assertIn("u1652_D2S/index.html", archive.namelist())
                self.assertIn("u1652_S2D/queries.json", archive.namelist())
                self.assertFalse(any(p.endswith(".pth") for p in archive.namelist()))

    def test_failed_training_does_not_shutdown_or_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(run_experiments, "ROOT", root), patch("sys.argv", ["runner", "--name", "failure", "--datasets", "U1652", "--shutdown"]), patch.object(run_experiments.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "train")) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    run_experiments.main()
            self.assertEqual(run.call_count, 1)
            status = json.loads((root / "outputs/failure/run.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertFalse((root / "outputs/failure/results.zip").exists())
