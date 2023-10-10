#!/bin/bash

perceptron_qubits=2
pulse_basis=2

total_control_fields=$((2 * perceptron_qubits))


for k in $(seq 0 $((total_control_fields-1)))  # Use 'seq' for numerical sequences
do
    for n in $(seq 0 $((pulse_basis-1)))  # Use 'seq' for numerical sequences
    do
        # Replace the following line with your command using k and n
        python full_rank_script.py --qubits "$perceptron_qubits" --field 0.1 --seed 92183 --basis "$pulse_basis" --sigma 5e-2 --k "$k" --n "$n" -outdir 'Results/5e-2sigma'

        # Print the output for each combination of k and n
        echo "Command executed with k = $k, n = $n"
    done
done

echo "Script finished."
