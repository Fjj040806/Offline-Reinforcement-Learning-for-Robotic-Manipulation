"""Export inference weights without optimizer state, for compact distribution."""
from pathlib import Path
from flax import serialization
root=Path(__file__).resolve().parents[1]
for p in list((root/'results').glob('*/agent.msgpack'))+list((root/'ablations').glob('*/agent.msgpack')):
 if not (p.parent/'summary.json').exists():continue
 d=serialization.msgpack_restore(p.read_bytes());(p.parent/'weights.msgpack').write_bytes(serialization.msgpack_serialize({'params':d['network']['params'],'rng':d['rng']}))
