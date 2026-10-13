"""Pinned OGBench algorithms; explicit CPU budget overrides; local-only logging."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys,argparse,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/impls'))
import numpy as np,jax,jax.numpy as jnp,ogbench
from flax import serialization
from agents import gcbc,gciql,hiql
from utils.datasets import Dataset,GCDataset,HGCDataset

p=argparse.ArgumentParser();p.add_argument('--algorithm',choices=['gcbc','gciql','hiql'],default='gcbc');p.add_argument('--dataset',default='cube-single-play-v0');p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=20000);p.add_argument('--episodes',type=int,default=50);p.add_argument('--width',type=int,default=128);p.add_argument('--batch',type=int,default=256);p.add_argument('--data-dir',default=str(ROOT/'data'));p.add_argument('--reset-start',type=int,default=10000);p.add_argument('--restore',action='store_true');p.add_argument('--fraction',type=float,default=1.0);p.add_argument('--subgoal',type=int,default=10);p.add_argument('--out',default=None);a=p.parse_args()
np.random.seed(a.seed);out=Path(a.out) if a.out else ROOT/'results'/f'{a.dataset}-{a.algorithm}-s{a.seed}';out.mkdir(parents=True,exist_ok=True)
# Serialize concurrent invocations targeting the same output directory.
import fcntl
lock=(out/'.run.lock').open('w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if (out/'summary.json').exists():print('Already completed by another worker',flush=True);sys.exit(0)
mod={'gcbc':gcbc,'gciql':gciql,'hiql':hiql}[a.algorithm];cls={'gcbc':gcbc.GCBCAgent,'gciql':gciql.GCIQLAgent,'hiql':hiql.HIQLAgent}[a.algorithm]
cfg=mod.get_config();cfg.encoder=None;cfg.frame_stack=None;cfg.batch_size=a.batch;cfg.actor_hidden_dims=(a.width,a.width)
if 'value_hidden_dims' in cfg:cfg.value_hidden_dims=(a.width,a.width)
if a.algorithm=='gciql':cfg.alpha=1.0
if a.algorithm=='hiql':cfg.subgoal_steps=a.subgoal
(out/'config.json').write_text(json.dumps({'arguments':vars(a),'agent':cfg.to_dict(),'upstream_commit':'a977253ce5b88434f4e4f531c50047cc95993a51'},indent=2))
env,train,val=ogbench.make_env_and_datasets(a.dataset,dataset_dir=a.data_dir,compact_dataset=True,width=320,height=240)
if not 0<a.fraction<=1:raise ValueError('fraction must be in (0,1]')
if a.fraction<1:
 raw_ends=np.flatnonzero(train['valids']==0);starts=np.r_[0,raw_ends[:-1]+1];ids=np.sort(np.random.default_rng(42).choice(len(raw_ends),max(1,int(len(raw_ends)*a.fraction)),replace=False));indices=np.concatenate([np.arange(starts[k],raw_ends[k]+1) for k in ids]);train={k:v[indices] for k,v in train.items()};(out/'trajectory_subset.json').write_text(json.dumps(ids.tolist()))
manifest=json.loads((ROOT/'docs/data_manifest.json').read_text())
for name in [a.dataset+'.npz',a.dataset+'-val.npz']:
 h=hashlib.sha256()
 with (Path(a.data_dir)/name).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 if h.hexdigest()!=manifest[name]['sha256']:raise ValueError('Dataset SHA256 mismatch: '+name)
dc=HGCDataset if a.algorithm=='hiql' else GCDataset;tr=dc(Dataset.create(**train),cfg);va=dc(Dataset.create(**val),cfg);b=tr.sample(1);agent=cls.create(a.seed,b['observations'],b['actions'],cfg);checkpoint=out/'agent.msgpack';t=time.time();curve=[]
if a.restore:
 if checkpoint.exists():agent=serialization.from_bytes(agent,checkpoint.read_bytes())
 else:
  weights=serialization.msgpack_restore((out/'weights.msgpack').read_bytes());agent=agent.replace(network=agent.network.replace(params=serialization.from_state_dict(agent.network.params,weights['params'])),rng=weights['rng'])
 old_curve=json.loads((out/'learning_curve.json').read_text());t-=old_curve[-1]['seconds']
else:
 for step in range(1,a.steps+1):
  agent,metrics=agent.update(tr.sample(a.batch))
  if step%1000==0 or step==a.steps:
   jax.block_until_ready(agent.network.params);_,v=agent.total_loss(va.sample(a.batch),grad_params=None);row={'step':step,'seconds':time.time()-t,'train':{k:float(x) for k,x in metrics.items()},'validation':{k:float(x) for k,x in v.items()}};curve.append(row);print(a.algorithm,a.dataset,a.seed,step,round(row['seconds'],1),flush=True)
 (out/'learning_curve.json').write_text(json.dumps(curve,indent=2));tmp=out/'agent.tmp';tmp.write_bytes(serialization.to_bytes(agent));tmp.replace(checkpoint)
rows=[];replays=[];rng=jax.random.PRNGKey(a.seed+1000)
# Fixed evaluation starts shared by all algorithms and training seeds.
for task in range(1,6):
 for ep in range(a.episodes):
  ob,info=env.reset(seed=a.reset_start+ep,options={'task_id':task});goal=info['goal'];done=False;steps=0;trajectory=[]
  while not done:
   rng,key=jax.random.split(rng);act=np.asarray(agent.sample_actions(observations=jnp.asarray(ob),goals=jnp.asarray(goal),seed=key,temperature=0.0));ob,r,terminated,truncated,info=env.step(act);steps+=1;done=bool(terminated or truncated)
   if ep==0 and (steps%10==0 or done):
    d=env.unwrapped.compute_ob_info();trajectory.append({'step':steps,'effector':d['proprio/effector_pos'].tolist(),'blocks':[d[f'privileged/block_{k}_pos'].tolist() for k in range(1 if 'single' in a.dataset else 2)]})
  rows.append({'task':task,'reset_seed':a.reset_start+ep,'steps':steps,'success':bool(info['success']),'terminated':bool(terminated),'truncated':bool(truncated)})
  if ep==0:replays.append({'task':task,'reset_seed':a.reset_start,'success':bool(info['success']),'goal':goal.tolist(),'frames':trajectory})
 print('EVAL',a.algorithm,a.dataset,a.seed,task,sum(x['success'] for x in rows if x['task']==task),'/',a.episodes,flush=True)
 (out/'episodes.json').write_text(json.dumps(rows))
summary={'algorithm':a.algorithm,'dataset':a.dataset,'seed':a.seed,'steps':a.steps,'episodes':len(rows),'successes':sum(x['success'] for x in rows),'success_rate':float(np.mean([x['success'] for x in rows])),'per_task':{str(k):float(np.mean([x['success'] for x in rows if x['task']==k])) for k in range(1,6)},'total_seconds':time.time()-t,'devices':[str(x) for x in jax.devices()],'fraction':a.fraction,'subgoal_steps':a.subgoal,'reset_start':a.reset_start,'network_width':a.width,'batch_size':a.batch}
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'replays.json').write_text(json.dumps(replays));env.close();print('COMPLETE',json.dumps(summary),flush=True)
