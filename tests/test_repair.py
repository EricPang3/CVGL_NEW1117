"""修复只在原队列结束后应用，旧成功结果和失败现场可追溯。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.repair_experiments import prepare_repair


class RepairTests(unittest.TestCase):
    def setup_run(self, root, status):
        config={'run':'run','modules':[{'id':'danet_pam'},{'id':'eca'}]}
        state={'status':status,'jobs':{},'results':[],'events':[]}
        p=root/'outputs/run';(p/'archive_ready').mkdir(parents=True)
        for module,label in (('danet_pam','u1652'),('danet_pam','sues200_150'),('eca','u1652')):
            job=f'{module}_{label}'
            state['jobs'][job]={'module':module,'dataset':'U1652' if label=='u1652' else 'SUES200',
                                'altitude':None if label=='u1652' else 150,'completed':['train'],
                                'status':'failed' if job=='danet_pam_u1652' else 'complete'}
            state['results'].append({'module':module,'direction':'D2S'})
            model=root/'outputs'/f'run_{job}';model.mkdir();(model/'config.yaml').write_text('old recipe')
            (p/'archive_ready'/f'{job}.json').write_text('{}')
        (p/'state.json').write_text(json.dumps(state))
        return config,state,{'id':'pam_fp32','modules':['danet_pam'],'reason':'AMP stability','validated_commit':'test'}

    def test_repair_preserves_old_records_and_resets_all_selected_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,before,request=self.setup_run(root,'partial_failure')
            self.assertTrue(prepare_repair(root,config,request))
            p=root/'outputs/run';state=json.loads((p/'state.json').read_text())
            self.assertEqual(state['status'],'repair_pending')
            self.assertEqual(state['jobs']['eca_u1652'],before['jobs']['eca_u1652'])
            self.assertTrue(all(r['module']=='eca' for r in state['results']))
            for job in ('danet_pam_u1652','danet_pam_sues200_150'):
                self.assertEqual(state['jobs'][job]['completed'],[])
                self.assertTrue((root/'outputs'/f'run_{job}_before_pam_fp32/config.yaml').exists())
                self.assertFalse((p/'archive_ready'/f'{job}.json').exists())
            saved=json.loads((p/'repairs/pam_fp32/state_before.json').read_text())
            self.assertEqual(saved,before)
            self.assertFalse(prepare_repair(root,config,request))

    def test_repair_cannot_mutate_running_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,before,request=self.setup_run(root,'running')
            with self.assertRaises(RuntimeError):prepare_repair(root,config,request)
            self.assertEqual(json.loads((root/'outputs/run/state.json').read_text()),before)
            self.assertTrue((root/'outputs/run_danet_pam_u1652/config.yaml').exists())
