
#!/bin/bash
#SBATCH -J analog_perceptron_4_qubit_55_pulses_100_tmodel
#SBATCH -c 1 # Number of cores
#SBATCH -p shared
#SBATCH --mem 16000
#SBATCH -t 1-0:0 # Maximum execution time (D-HH:MM)
#SBATCH -o '/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/RESULTS/4_qubits/55_pulses_100_tmodel/perceptron_%A_%a.out' # Standard output
#SBATCH -e '/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/RESULTS/4_qubits/55_pulses_100_tmodel/perceptron_%A_%a.err' # Standard error
#SBATCH --array=1-10  # Size of the array

conda activate perceptron-loss-landscapes-venv

mkdir -p "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/RESULTS/4_qubits/55_pulses_100_tmodel"

mkdir -p "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/RESULTS/4_qubits/55_pulses_100_tmodel/${SLURM_ARRAY_TASK_ID}"

cd "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/"

python "/n/home08/jgarciaponce/Yelin/perceptron-loss-landscapes/Evolution Time Experiments/full_perceptron.py" --qubits 4 --pulses 55 --model_time 100 --loss_time 1 --learning_rate 0.01 --save_path RESULTS/4_qubits/55_pulses_100_tmodel/${SLURM_ARRAY_TASK_ID}/55_pulses_100_tmodel_simulation_data.pickle
