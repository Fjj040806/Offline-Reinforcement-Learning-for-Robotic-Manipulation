import json,unittest,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class EvidenceTests(unittest.TestCase):
 def test_complete_matrix(self):
  paths=list((ROOT/'results').glob('*/summary.json'));self.assertEqual(len(paths),18)
  for p in paths:
   s=json.loads(p.read_text());eps=json.loads((p.parent/'episodes.json').read_text());self.assertEqual(len(eps),250);self.assertEqual(s['successes'],sum(e['success'] for e in eps));self.assertEqual(s['success_rate'],s['successes']/250)
   self.assertEqual({e['reset_seed'] for e in eps},set(range(10000,10050)))
   self.assertEqual({e['task'] for e in eps},{1,2,3,4,5})
 def test_replay_evidence(self):
  for p in (ROOT/'results').glob('*/replays.json'):
   rs=json.loads(p.read_text());self.assertEqual(len(rs),5)
   for r in rs:
    self.assertTrue(r['frames']);self.assertEqual(r['reset_seed'],10000)
    self.assertEqual(sorted(f['step'] for f in r['frames']),[f['step'] for f in r['frames']])
 def test_ablation_matrix(self):
  paths=list((ROOT/'ablations').glob('*/summary.json'));self.assertEqual(len(paths),24)
  subsets={}
  for p in paths:
   summary=json.loads(p.read_text());episodes=json.loads((p.parent/'episodes.json').read_text());self.assertEqual(len(episodes),50);self.assertEqual(summary['successes'],sum(x['success'] for x in episodes));self.assertEqual({x['reset_seed'] for x in episodes},set(range(20000,20010)))
   manifest=p.parent/'trajectory_subset.json'
   if manifest.exists():
    ids=json.loads(manifest.read_text());key=summary['fraction'];self.assertEqual(len(ids),int(key*1000))
    if key in subsets:self.assertEqual(ids,subsets[key])
    subsets[key]=ids
 def test_training_metrics_finite(self):
  for p in (ROOT/'results').glob('*/learning_curve.json'):
   for point in json.loads(p.read_text()):
    for section in ['train','validation']:
     self.assertTrue(all(math.isfinite(v) for v in point[section].values()))
 def test_all_summaries_finite(self):
  for p in (ROOT/'results').glob('*/summary.json'):
   s=json.loads(p.read_text());self.assertTrue(0<=s['success_rate']<=1)
if __name__=='__main__':unittest.main()
