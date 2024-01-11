# !/bin/sh

python3 experiments/PDE/cutoff.py
python3 experiments/PDE/timestep.py
python3 experiments/PDE/dof.py
python3 experiments/SDE/timestep.py

python3 experiments/Variance_reduction/variance.py
python3 experiments/run_SDE.py