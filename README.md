# Case Study 1 — Precision Trajectory Tracking (PyBullet-Drones)

## Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Run one seed
python -m src.run_one --config configs/scenario_default.yaml --controller openloop
python -m src.run_one --config configs/scenario_default.yaml --controller pid
python -m src.run_one --config configs/scenario_default.yaml --controller agentic

## Run batch and generate plots
python -m src.run_all --config configs/scenario_default.yaml --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

Outputs:
- outputs/csv/*.csv
- outputs/figs/*.png
