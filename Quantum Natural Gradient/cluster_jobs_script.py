import numpy as np
import subprocess

# Define the range and number of steps for LEARNING_RATE
start_lr = 0.01  # start value
end_lr = 5   # end value
num_steps = 30  # number of steps

# Generate LEARNING_RATE values
learning_rates = np.linspace(start_lr, end_lr, num_steps)

# Constants
QUBITS = 4
BASIS = 8
DAYS = 0
HOURS = 6
MINUTES = 0

# Template for the filename
filename_template = 'perceptron_{qubits}_qubits_{basis}_pulses_{lr}_lr.sh'

# Loop over each LEARNING_RATE value
for lr in learning_rates:
    # Prepare the filename with the current LEARNING_RATE
    filename = filename_template.format(qubits=QUBITS, basis=BASIS, lr=lr)

    # Prepare the bash command for envsubst
    bash_command = f'export QUBITS={QUBITS} BASIS={BASIS} LEARNING_RATE={lr} DAYS={DAYS} HOURS={HOURS} MINUTES={MINUTES} && envsubst \'$QUBITS, $BASIS, $LEARNING_RATE, $DAYS, $HOURS, $MINUTES\' < BATCH_TEMPLATE.sh > {filename}'

    # Execute the envsubst command to create the .sh file
    subprocess.run(bash_command, shell=True, check=True)

    # Now, run sbatch for the created file
    sbatch_command = f'sbatch {filename}'
    subprocess.run(sbatch_command, shell=True, check=True)

    print(f"Submitted batch job for {filename}")

print("All jobs have been submitted.")
