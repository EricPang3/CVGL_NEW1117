"""记录百分比、配对seed和未运行项；只有完整结果已留存时才能删权重。"""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.experiment_records import export_tables, keep_checkpoint, prune_local


class ExperimentRecordTests(unittest.TestCase):
    def test_tables_keep_seed_pairing_and_blank_unrun_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan=root/'outputs/overnight20261120';plan.mkdir(parents=True)
            (plan/'state.json').write_text(json.dumps({'runs':['seed1','seed2']}))
            for seed,score,delta in ((1,92.5,.1),(2,92.0,-.3)):
                p=root/'outputs'/f'seed{seed}';p.mkdir()
                row={'module':'eca','dataset':'U1652','altitude':None,'direction':'D2S','recall@1':score,'mAP':93.1,
                     'delta_R1_pp':delta,'delta_AP_pp':.2}
                rows=[row,{**row,'module':'wtconv'}] if seed==1 else [row]
                (p/'state.json').write_text(json.dumps({'seed':seed,'status':'running','results':rows}))
            records=export_tables(root)
            baseline=[r for r in records if r['module']=='baseline' and r['seed']==2][0]
            self.assertAlmostEqual(baseline['recall@1'],92.3)
            with (root/'results/seed2_R1.csv').open(encoding='utf-8-sig') as stream:
                wide={r['module']:r for r in csv.DictReader(stream)}
            self.assertEqual(wide['eca']['U1652_NA_D2S'],'92.0')
            self.assertEqual(wide['eca']['SUES200_150_D2S'],'')
            self.assertEqual(wide['wtconv']['completed_evaluations'],'0')
            with (root/'results/replication.csv').open(encoding='utf-8-sig') as stream:pair=list(csv.DictReader(stream))[1]
            self.assertEqual(pair['module'],'eca')
            self.assertEqual(float(pair['seed1_delta_R1_pp']),.1)
            self.assertEqual(float(pair['repeat_delta_R1_pp']),-.3)

    def setup_artifacts(self, root):
        p=root/'outputs/modules20261120';p.mkdir(parents=True)
        cfg=root/'configs';cfg.mkdir()
        policy={'eligible_modules':['eca','wtconv'],'keep':[{'module':'wtconv','dataset':'U1652','altitude':None,'reason':'control'}]}
        (cfg/'checkpoint_retention.json').write_text(json.dumps(policy))
        state={'status':'running','jobs':{},'results':[]};manifest={'artifacts':{}}
        for job,module,alt,status in [('wtconv_u1652','wtconv',None,'complete'),('eca_sues200_200','eca',200,'complete'),('eca_sues200_150','eca',150,'running')]:
            model=p/'artifacts'/job;model.mkdir(parents=True)
            for name in ('model.pth','config.yaml','train.csv','source.zip'):(model/name).write_bytes(b'keep records')
            for d in ('D2S','S2D'):
                q=model/d;q.mkdir()
                for name in ('metrics.json','queries.json','report.zip'):(q/name).write_text('{}')
            state['jobs'][job]={'module':module,'dataset':'U1652' if alt is None else 'SUES200','altitude':alt,'status':status}
            manifest['artifacts'][job]={'job':job,'local_directory':str(model),'archive_checkpoint':True,'checkpoint_sha256':hashlib.sha256(b'keep records').hexdigest()}
        (p/'state.json').write_text(json.dumps(state));(p/'archive_manifest.json').write_text(json.dumps(manifest))
        return p,policy

    def test_cleanup_preserves_records_selected_and_running_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,policy=self.setup_artifacts(root)
            result=prune_local(root)
            self.assertEqual(result['deleted_weights'],1)
            self.assertTrue((p/'artifacts/wtconv_u1652/model.pth').exists())
            self.assertTrue((p/'artifacts/eca_sues200_150/model.pth').exists())
            self.assertFalse((p/'artifacts/eca_sues200_200/model.pth').exists())
            self.assertTrue((p/'artifacts/eca_sues200_200/D2S/queries.json').exists())
            self.assertTrue((root/'results/all_metrics.csv').exists())
            self.assertEqual(prune_local(root)['deleted_weights'],0)

    def test_incomplete_result_blocks_weight_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,_=self.setup_artifacts(root)
            (p/'artifacts/eca_sues200_200/D2S/queries.json').unlink()
            with self.assertRaises(RuntimeError):prune_local(root)
            self.assertTrue((p/'artifacts/eca_sues200_200/model.pth').exists())

    def test_policy_protects_baseline_and_unknown_future_modules(self):
        policy={'eligible_modules':['eca'],'keep':[]}
        self.assertTrue(keep_checkpoint(policy,{'module':'eca'},{'job':'baseline_none_u1652'}))
        self.assertTrue(keep_checkpoint(policy,{'module':'future'},{'job':'future_u1652'}))
        self.assertFalse(keep_checkpoint(policy,{'module':'eca','dataset':'U1652','altitude':None},{'job':'eca_u1652'}))


if __name__=='__main__':unittest.main()
