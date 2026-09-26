# scripts/analysis: offline theory checks

Both scripts recompute numbers from stored attack systems (`*_paired_probe.npz`: `A`, `G`, `true_U`,
`cos`, `attacked_uids`, `eval_mask`) and from warm-up checkpoints; no training. Paths: repository
root = parent of `scripts/`; results root `TRIP_RESULTS` (default `<repo>/results`); datasets
`FL_DATA_ROOT` (default `<repo>/data_local`); warm-up cache `FL_WARMUP_CACHE` (default
`<repo>/warmup_cache`). Outputs go to `<results>/analysis/`.

## `offline_checks.py [section ...]` -> `analysis/offline.json`

Default: all sections except `sat`. Sections already present in the output file are kept.

- `proj`: measured cosine vs. the projection ceiling cos(PU, U), P = orthogonal projector onto row(A),
  on the ten MF W=20 runs in `results/main`; also rank(A), per-user max deviation, norm CV.
- `lgcn`: LightGCN target mismatch: cosine between the propagated target
  u~ = (2 u0 + sum_items v / sqrt(n)) / 3 and the stored u0, vs. TRIP W=21 (`results/v3/headline_w21`,
  item embeddings from the warm-up cache).
- `noise`: effective gains K W ||row_i(A_lambda^+)|| of the window operator (N=500, K=10, T=150,
  lambda=1e-6) for W in {19, 20, 21, 25}, 1/sigma_min and rank, and the Prop. 2 prediction at
  sigma=1e-6 (200 Monte-Carlo draws) vs. the measured `results/biasvar/W<W>_n1e-06` LastFM/MF run.
- `mixed`: rank and 1/sigma_min of the W=20, W=21 and mixed (100 rounds W=20 + 50 rounds W=21) schedules.
- `ladder`: the Adam threat-model rungs (`results/threatmodel/tm_*_W2[01]`, replaced by
  `results/v3/threatmodel/tm3_*` where present) re-solved with K per-pair offset columns: plain vs.
  offset cosine, agreement with sign(U), stored cosine, null, relative residuals.
- `colspace`: share of G outside col(A) (relative residual of lstsq(A, G)) for every stored W20/W21
  cell in `results/{v3,v2,}/{main,headline_w21}` (first existing per cell).
- `rounds`: K W > N schedules (`configs/rounds_K25_Tf1.yaml`, `rounds_K50_Tf1.yaml`): the weighted
  operator rebuilt from the allocator, its rank and projection ceiling vs. the measured
  `results/v2/rounds/<cfg>` runs, and the largest per-round multiplicity.
- `sat`: share of users whose honest observed update is ~0 (BPR saturated) on the `configs/default.yaml`
  checkpoints (500 targets, 500 ordinary users), plus InvGrad's cosine split by zero / non-zero update
  (`results/v3/main`, `results/main`). Needs a GPU and the warm-up cache.

## `anon_ceiling.py` -> `analysis/anon_ceiling.json`

Theorem-2 anonymity ceiling from the true embeddings of the W=21 TRIP runs (`results/v3/headline_w21`,
`results/v2/headline_w21`, `results/headline_w21`; first existing per cell). For block sizes
t in {1, 2, 4, 8, 16, 32, 64} and both an unstratified and an activity-stratified beacon partition
(seed `pact-v1`) of the N targets, it reports the mean over targets of ||m_Gamma|| (m_Gamma = mean unit
embedding of the target's block), the block-size range and block count, plus the null cosine and the
best constant predictor. CPU only; needs the datasets for the activity strata.
`scripts/v4/summarize_v4.py tbnd` reads this file.
