#!/bin/bash
#SBATCH -J analog_perceptron_4_qubit_7_pulses_1_tmodel
#SBATCH -c 1 # Number of cores
#SBATCH -p shared
#SBATCH --mem 16000
#SBATCH -t 0-6:0 # Maximum execution time (D-HH:MM)
#SBATCH -o '/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Full Perceptron Hamiltonian Experiments/RESULTS/4_qubits_MANY/7_pulses_1_tmodel/perceptron_%A_%a.out' # Standard output
#SBATCH -e '/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Full Perceptron Hamiltonian Experiments/RESULTS/4_qubits_MANY/7_pulses_1_tmodel/perceptron_%A_%a.err' # Standard error
#SBATCH --array=1-50  # Size of the array

conda activate perceptron-loss-landscapes-venv

mkdir -p "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Full Perceptron Hamiltonian Experiments/RESULTS/4_qubits_MANY/7_pulses_1_tmodel"

mkdir -p "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Full Perceptron Hamiltonian Experiments/RESULTS/4_qubits_MANY/7_pulses_1_tmodel/${SLURM_ARRAY_TASK_ID}"

cd "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Full Perceptron Hamiltonian Experiments/"

python "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Full Perceptron Hamiltonian Experiments/full_perceptron_hessians.py" --qubits 4 --pulses 7 --model_time 1 --loss_time 1 --learning_rate 0.1 --save_path RESULTS/4_qubits_MANY/7_pulses_1_tmodel/${SLURM_ARRAY_TASK_ID}/7_pulses_1_tmodel_simulation_data.pickle
