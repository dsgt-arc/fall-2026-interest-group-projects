"""Stage 3 (after the pilot regression): opening chunk in the last slot instead of the first.
Selected on dev + pilot; test is reported again and that reuse is disclosed in the README."""

from common import RESULTS, save_json, open_index
from engine import Engine
from experiments import BASE, HEADER, cases, line, run

index, models = open_index()
engine = Engine(index, models)
grid = [
    BASE,
    BASE.but(name="named intro + exact", intro="named", dense_search="exact"),
    BASE.but(name="intro last + exact", intro="last", dense_search="exact"),
    BASE.but(name="intro last + exact + RRF 0.3/0.7", intro="last", dense_search="exact", w_dense=0.3, w_bm25=0.7),
]
out = {}
for split in ("dev", "pilot", "test"):
    print(f"\n## {split}\n" + HEADER)
    out[split] = {}
    for cfg in grid:
        agg, _ = run(engine, cfg, cases(split), index)
        out[split][cfg.name] = agg
        print(line(cfg.name, agg), flush=True)
save_json(RESULTS / "intro_placement.json", out)
