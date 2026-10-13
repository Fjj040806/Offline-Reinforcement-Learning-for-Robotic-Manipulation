"""Three-seed controlled ablations; 10 development resets per task, separate from core tests."""
import subprocess,sys,os,concurrent.futures,shutil,time,argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--retrain',action='store_true');flags=parser.parse_args()
root=Path(__file__).resolve().parents[1];cpus=sorted(os.sched_getaffinity(0));jobs=[]
for seed in [0,1,2]:
 for alg in ['gciql','hiql']:
  for fraction in [.25,.5,1.]:jobs.append((alg,seed,fraction,10))
 for k in [5,25]:jobs.append(('hiql',seed,1.,k))
jobs.sort(key=lambda j: j[2]==1. and j[3]==10)
def work(worker):
 for i,(alg,seed,fraction,k) in enumerate(jobs):
  if i%workers!=worker:continue
  out=root/'ablations'/f'{alg}-s{seed}-f{fraction}-k{k}';out.mkdir(parents=True,exist_ok=True)
  if (out/'summary.json').exists() and not flags.retrain:continue
  if fraction==1. and k==10:
   base=root/'results'/f'cube-double-play-v0-{alg}-s{seed}'
   while not (base/'summary.json').exists():time.sleep(2)
   model='agent.msgpack' if (base/'agent.msgpack').exists() else 'weights.msgpack'
   if model=='weights.msgpack' and (out/'agent.msgpack').exists():(out/'agent.msgpack').unlink()
   for name in [model,'learning_curve.json']:shutil.copy2(base/name,out/name)
  args=['taskset','-c',','.join(map(str,cpus[worker*2:worker*2+2])),sys.executable,str(root/'scripts/run.py'),'--dataset','cube-double-play-v0','--algorithm',alg,'--seed',str(seed),'--steps','20000','--episodes','10','--reset-start','20000','--fraction',str(fraction),'--subgoal',str(k),'--out',str(out)]
  if (fraction==1. and k==10) or (not flags.retrain and ((out/'agent.msgpack').exists() or (out/'weights.msgpack').exists())):args.append('--restore')
  with (out/'run.log').open('w') as f:subprocess.run(args,check=True,stdout=f,stderr=subprocess.STDOUT)
  print('ABLATION DONE',alg,seed,fraction,k,flush=True)
workers=min(2,max(1,len(cpus)//2))
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:list(ex.map(work,range(workers)))
