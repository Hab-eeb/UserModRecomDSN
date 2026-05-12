# DSN x BCT Hackathon Baseline

Run dataset preparation from the repository root:

```bash
python src/recagent/data/prepare_dataset.py --config configs/baseline.yaml --skip-meta
```

The codebase uses a `src/` layout, and the entry scripts are set up to add `src` to the import path when run directly.
