"""Locked core experiment matrix, three independent CPU process workers."""
import subprocess,sys,os,concurrent.futures,argparse
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--retrain',action='store_true');args=ap.parse_args()
root=Path(__file__).resolve().parents[1];cpus=sorted(os.sched_getaffinity(0));env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
# Download before starting workers, preventing partial concurrent dataset downloads.
for dataset in ['cube-single-play-v0','cube-double-play-v0']:
 subprocess.run([sys.executable,'-c',f"import ogbench; e,t,v=ogbench.make_env_and_datasets({dataset!r},dataset_dir={str(root/'data')!r},compact_dataset=True);e.close()"],check=True,env=env)
jobs=[(d,a,s) for d in ['cube-single-play-v0','cube-double-play-v0'] for a in ['gcbc','gciql','hiql'] for s in [0,1,2]]
def work(worker):
 for i,(d,a,s) in enumerate(jobs):
  if i%workers!=worker:continue
  out=root/'results'/f'{d}-{a}-s{s}'
  if (out/'summary.json').exists() and not args.retrain:continue
  cmd=['taskset','-c',','.join(map(str,cpus[worker*2:worker*2+2])),sys.executable,str(root/'scripts/run.py'),'--dataset',d,'--algorithm',a,'--seed',str(s),'--steps','20000','--episodes','50']
  if not args.retrain and ((out/'agent.msgpack').exists() or (out/'weights.msgpack').exists()):cmd.append('--restore')
  with (root/'results'/f'{d}-{a}-s{s}.log').open('w') as log:subprocess.run(cmd,check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
  print('DONE',d,a,s,flush=True)
workers=min(3,max(1,len(cpus)//2))
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:list(ex.map(work,range(workers)))
