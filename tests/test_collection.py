"""校验成功才清理、限定本批次路径、清理可重试、完成日志可追溯。"""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.collect_experiments import REMOTE_ACK, REMOTE_POLL, begin_revision


class CollectionTests(unittest.TestCase):
    def test_retrained_artifact_preserves_previous_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'artifacts/pam_u1652';target.mkdir(parents=True)
            (target/'model.pth').write_bytes(b'old valid checkpoint')
            previous={'job':'pam_u1652','bundle_sha256':'a'*64,'checkpoint_sha256':'1'*64,'local_directory':str(target)}
            manifest={'artifacts':{'pam_u1652':previous}}
            self.assertFalse(begin_revision(root,manifest,previous))
            updated={**previous,'bundle_sha256':'b'*64,'checkpoint_sha256':'2'*64}
            self.assertTrue(begin_revision(root,manifest,updated))
            old=root/'history/pam_u1652'/('a'*64)
            self.assertEqual((old/'model.pth').read_bytes(),b'old valid checkpoint')
            self.assertNotIn('pam_u1652',manifest['artifacts'])
            self.assertEqual(len(manifest['artifact_history']['pam_u1652']),1)
            self.assertTrue(begin_revision(root,manifest,updated))
            self.assertEqual(len(manifest['artifact_history']['pam_u1652']),1)

    def test_cleanup_checks_hash_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "run_eca"
            model.mkdir()
            (model/"model.pth").write_bytes(b"verified weight")
            (model/"D2S/report").mkdir(parents=True)
            (model/"D2S/report/index.html").write_text("report")
            (model/"D2S/queries.json").write_text("{}")
            (model/"D2S/metrics.json").write_text("{}")
            bundle = root/"task.tar.gz"
            bundle.write_bytes(b"verified bundle")
            code = REMOTE_ACK.replace("Path('/root/autodl-tmp/CVGL_NEW/outputs')",f"Path({tmp!r})")
            args = {"job":"eca", "run":"run", "model_dir":str(model),"checkpoint":str(model/"model.pth"),
                    "bundle":str(bundle),"archive_checkpoint":True,"retain_remote":False,"checkpoint_sha256":"incorrect"}
            wrong = subprocess.run([sys.executable,"-c",code,json.dumps(args)],capture_output=True)
            self.assertNotEqual(wrong.returncode,0)
            self.assertTrue((model/"model.pth").exists())
            args["checkpoint_sha256"] = hashlib.sha256(b"verified weight").hexdigest()
            for _ in range(2):
                subprocess.run([sys.executable,"-c",code,json.dumps(args)],check=True,capture_output=True)
            self.assertFalse((model/"model.pth").exists())
            self.assertFalse((model/"D2S/queries.json").exists())
            self.assertTrue((model/"D2S/metrics.json").exists())
            self.assertTrue((model/"ARCHIVED_LOCALLY.json").exists())

    def test_poll_includes_completed_log_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/"run";run.mkdir()
            (run/"state.json").write_text(json.dumps({"status":"running","jobs":{"eca":{"status":"complete"}}}))
            (run/"eca.log").write_text("training and evaluation logs")
            code=REMOTE_POLL.replace("Path('/root/autodl-tmp/CVGL_NEW/outputs')",f"Path({tmp!r})")
            result=json.loads(subprocess.check_output([sys.executable,"-c",code,"run"],text=True))
            self.assertEqual(result["logs"][0]["sha256"],hashlib.sha256(b"training and evaluation logs").hexdigest())
