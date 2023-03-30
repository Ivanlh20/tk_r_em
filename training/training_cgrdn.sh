# HRSEM
python -O ./nn_training.py 4 119 3 1.0e-6 0.9 0 0.99 'relu' 4 0.50 2 1e-8 1e-7 0 2e-8 0 0.001 1.0 0.01 01 0 0.5 4 &

# SEM
python -O ./nn_training.py 5 219 3 1.0e-6 0.9 0 0.99 'relu' 4 0.50 2 1e-8 1e-7 0 2e-8 0 0.001 1.0 0.01 01 0 0.5 4 &

# HR-STEM
python -O ./nn_training.py 0 319 3 2.5e-9 0.5 0 0.99 'relu' 4 0.50 0 1e-7 1e-6 0 2e-8 0 0.001 1.0 0.01 01 0 0.5 4 &

# STEM
python -O ./nn_training.py 7 419 3 1.0e-6 0.9 0 0.99 'relu' 4 0.50 2 1e-8 1e-7 0 2e-8 0 0.001 1.0 0.01 01 0 0.5 4 &

# HR-TEM
python -O ./nn_training.py 8 519 3 1.0e-6 0.9 0 0.99 'relu' 4 0.50 2 1e-8 1e-7 0 2e-8 0 0.001 1.0 0.01 01 0 0.5 4 &

# TEM
python -O ./nn_training.py 9 619 3 1.0e-6 0.9 0 0.99 'relu' 4 0.50 2 1e-8 1e-7 0 2e-8 0 0.001 1.0 0.01 01 0 0.5 4 &