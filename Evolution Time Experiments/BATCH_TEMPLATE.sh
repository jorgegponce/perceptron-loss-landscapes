#!/bin/bash
#SBATCH -J analog_perceptron_${QUBITS}_qubit_${BASIS}_pulses_${TMODEL}_tmodel
#SBATCH -c 1 # Number of cores
#SBATCH -p shared
#SBATCH --mem 16000
#SBATCH -t ${DAYS}-${HOURS}:${MINUTES} # Maximum execution time (D-HH:MM)
#SBATCH -o '/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/RESULTS/${QUBITS}_qubits/${BASIS}_pulses_${TMODEL}_tmodel/perceptron_%A_%a.out' # Standard output
#SBATCH -e '/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/RESULTS/${QUBITS}_qubits/${BASIS}_pulses_${TMODEL}_tmodel/perceptron_%A_%a.err' # Standard error
#SBATCH --array=1-10  # Size of the array

conda activate perceptron-loss-landscapes-venv

mkdir -p "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/RESULTS/${QUBITS}_qubits/${BASIS}_pulses_${TMODEL}_tmodel"

mkdir -p "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/RESULTS/${QUBITS}_qubits/${BASIS}_pulses_${TMODEL}_tmodel/${SLURM_ARRAY_TASK_ID}"

cd "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/"

python "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/full_perceptron.py" --qubits ${QUBITS} --pulses ${BASIS} --model_time ${TMODEL} --loss_time 1 --learning_rate ${LR} --save_path RESULTS/${QUBITS}_qubits/${BASIS}_pulses_${TMODEL}_tmodel/${SLURM_ARRAY_TASK_ID}/${BASIS}_pulses_${TMODEL}_tmodel_simulation_data.pickle
