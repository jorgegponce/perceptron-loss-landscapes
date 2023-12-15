#!/bin/bash
#SBATCH -J analog_perceptron_${QUBITS}_qubit_${BASIS}_pulses
#SBATCH -c 1 # Number of cores
#SBATCH -p shared
#SBATCH --mem 16000
#SBATCH -t ${DAYS}-${HOURS}:${MINUTES} # Maximum execution time (D-HH:MM)
#SBATCH -o '/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Frobenius Norm Landscapes/Results/${QUBITS}_qubits/${BASIS}_pulses/perceptron_%A_%a.out' # Standard output
#SBATCH -e '/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Frobenius Norm Landscapes/Results/${QUBITS}_qubits/${BASIS}_pulses/perceptron_%A_%a.err' # Standard error
#SBATCH --array=1-2  # Size of the array

conda activate perceptron-loss-landscapes-venv

mkdir -p "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Frobenius Norm Landscapes/Results/${QUBITS}_qubits/${BASIS}_pulses"

cd "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Frobenius Norm Landscapes/Results/${QUBITS}_qubits/${BASIS}_pulses"

mkdir -p "${SLURM_ARRAY_TASK_ID}"

cd "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Frobenius Norm Landscapes"

python "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Frobenius Norm Landscapes/frobenius_norm_perceptron.py" --qubits ${QUBITS} --pulses ${BASIS} --save_path Results/${QUBITS}_qubits/${BASIS}_pulses/${SLURM_ARRAY_TASK_ID}/${BASIS}_pulses_simulation_data.pickle
