# Development Scripts (minimal)

This folder is trimmed for a personal project. Keep it simple and run only what you need.

## Available scripts

- `run_tests.sh` — run the core test suite (unit + integration) with optional coverage.
- `check_modular_separation.py` — optional static check to ensure data/statistics/plots/content stay decoupled.

## Usage

```bash
# from project root
./scripts/run_tests.sh             # run unit + integration + error-handling tests
./scripts/run_tests.sh --unit      # unit-only
./scripts/run_tests.sh --integration # integration-only
./scripts/run_tests.sh --coverage  # add coverage

# optional architecture check
python scripts/check_modular_separation.py
```

## Notes

- Requires `pytest` (install with `pip install -r requirements-dev.txt`).
- Set `PYTHONPATH` is handled inside the script; just run from project root.