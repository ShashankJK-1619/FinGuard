"""Portfolio visuals layer (FinGuard Portfolio Production, Step 3).

Pure presentation code: no new analytical modeling, no new metrics are
computed here. Every number drawn on a chart either comes from an
existing generated artifact under `data/interim/` (when present locally)
or from the centralized, source-cited fallback constants in
`metrics_config.py` (used when that artifact is gitignored and therefore
absent from a fresh clone). See `metrics_config.py` for the single
source-of-truth table and exact provenance of every figure used.
"""
