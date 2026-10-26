"""Stage 2 on dev only: combine the one-factor winners, then try gemma3:12b reranking on top."""

from common import RESULTS, save_json, open_index
from engine import Engine
from experiments import BASE, HEADER, cases, line, run

index, models = open_index()
engine = Engine(index, models)
rows = cases("dev")
named = BASE.but(intro="named", dense_search="exact")
grid = [
    BASE,
    named.but(name="named intro + exact"),
    named.but(name="named intro + exact + RRF 0.4/0.6", w_dense=0.4, w_bm25=0.6),
    named.but(name="named intro + exact + RRF 0.3/0.7", w_dense=0.3, w_bm25=0.7),
    named.but(name="named intro + exact + RRF 0.3/0.7 + dims 512", w_dense=0.3, w_bm25=0.7, dims=512),
    named.but(name="named intro + exact + RRF 0.3/0.7 + gemma rerank top 12", w_dense=0.3, w_bm25=0.7, rerank=12),
]
report = {}
print(HEADER)
for cfg in grid:
    agg, _ = run(engine, cfg, rows, index)
    report[cfg.name] = {"config": cfg.__dict__, "metrics": agg}
    print(line(cfg.name, agg), flush=True)
save_json(RESULTS / "dev_combinations.json", report)
