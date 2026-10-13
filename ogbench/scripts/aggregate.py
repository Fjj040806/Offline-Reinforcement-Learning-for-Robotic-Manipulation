"""Aggregate only complete real experiment runs; no imputed results."""
import json,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];summaries=[];replays=[];curves=[]
for path in sorted((root/'results').glob('*/summary.json')):
 s=json.loads(path.read_text());summaries.append(s)
 for r in json.loads((path.parent/'replays.json').read_text()):
  raw=r['frames'];r['frames']=[f for f in raw if f['step']%10==0];
  if raw and (not r['frames'] or r['frames'][-1]['step']!=raw[-1]['step']):r['frames'].append(raw[-1])
  replays.append(dict(r,algorithm=s['algorithm'],dataset=s['dataset'],seed=s['seed']))
 curves.append({'dataset':s['dataset'],'algorithm':s['algorithm'],'seed':s['seed'],'points':json.loads((path.parent/'learning_curve.json').read_text())})
groups=[]
for d in sorted({s['dataset'] for s in summaries}):
 for a in ['gcbc','gciql','hiql']:
  ss=[s for s in summaries if s['dataset']==d and s['algorithm']==a]
  if not ss:continue
  v=[s['success_rate'] for s in ss];groups.append({'dataset':d,'algorithm':a,'n_seeds':len(v),'mean':float(np.mean(v)),'seed_std':float(np.std(v,ddof=1)) if len(v)>1 else None,'episodes':sum(s['episodes'] for s in ss),'successes':sum(s['successes'] for s in ss)})
ablation_runs=[]
for p in sorted((root/'ablations').glob('*/summary.json')):
 r=json.loads(p.read_text());r['run_name']=p.parent.name;ablation_runs.append(r)
ablation_groups=[]
for key in sorted({(r['algorithm'],r['fraction'],r['subgoal_steps']) for r in ablation_runs}):
 rr=[r for r in ablation_runs if (r['algorithm'],r['fraction'],r['subgoal_steps'])==key];vals=[r['success_rate'] for r in rr];ablation_groups.append({'algorithm':key[0],'fraction':key[1],'subgoal_steps':key[2],'n_seeds':len(rr),'mean':float(np.mean(vals)),'seed_std':float(np.std(vals,ddof=1)) if len(vals)>1 else None,'episodes':sum(r['episodes'] for r in rr),'successes':sum(r['successes'] for r in rr)})
report={'ablation_runs':ablation_runs,'ablation_groups':ablation_groups,'scope':'CPU reduced-budget experiments, not convergence or published-benchmark reproduction','protocol':{'train_steps':20000,'hidden_dims':[128,128],'batch_size':256,'training_seeds':[0,1,2],'reset_start':10000,'episodes_per_task':50},'groups':groups,'runs':summaries,'curves':curves,'replays':replays}
(root/'docs/results.json').write_text(json.dumps(report));print('COMPLETE RUNS',len(summaries))
