
#!/bin/bash
#SBATCH -J analog_perceptron_4_qubit_25_pulses_1_tmodel
#SBATCH -c 1 # Number of cores
#SBATCH -p shared
#SBATCH --mem 16000
#SBATCH -t 0-2:0 # Maximum execution time (D-HH:MM)
#SBATCH -o '/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/RESULTS/4_qubits/25_pulses_1_tmodel/perceptron_%A_%a.out' # Standard output
#SBATCH -e '/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/RESULTS/4_qubits/25_pulses_1_tmodel/perceptron_%A_%a.err' # Standard error
#SBATCH --array=1-10  # Size of the array

conda activate perceptron-loss-landscapes-venv

mkdir -p "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/RESULTS/4_qubits/25_pulses_1_tmodel"

mkdir -p "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/RESULTS/4_qubits/25_pulses_1_tmodel/${SLURM_ARRAY_TASK_ID}"

cd "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/"

python "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/full_perceptron.py" --qubits 4 --pulses 25 --model_time 1 --loss_time 1 --learning_rate 0.1 --save_path RESULTS/4_qubits/25_pulses_1_tmodel/${SLURM_ARRAY_TASK_ID}/25_pulses_1_tmodel_simulation_data.pickle
