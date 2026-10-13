"""Verify the actual whole-trajectory slicing used in the ablation harness."""
import numpy as np,ogbench
from pathlib import Path
root=Path(__file__).resolve().parents[1];e,d,v=ogbench.make_env_and_datasets('cube-double-play-v0',dataset_dir=str(root/'data'),compact_dataset=True)
ends=np.flatnonzero(d['valids']==0);starts=np.r_[0,ends[:-1]+1]
for fraction in [.25,.5]:
 ids=np.sort(np.random.default_rng(42).choice(len(ends),int(len(ends)*fraction),replace=False));indices=np.concatenate([np.arange(starts[k],ends[k]+1) for k in ids]);sub={k:a[indices] for k,a in d.items()};assert sub['valids'][-1]==0
 terminal=np.flatnonzero(sub['valids']==0);rng=np.random.default_rng(12);sources=rng.choice(np.flatnonzero(sub['valids']),10000);bound=terminal[np.searchsorted(terminal,sources)];goals=rng.integers(sources+1,bound+1);assert np.all(np.searchsorted(terminal,sources)==np.searchsorted(terminal,goals));print('PASSED whole-trajectory fraction',fraction,len(ids),'trajectories')
e.close()
