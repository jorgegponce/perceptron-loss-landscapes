import time
import argparse
import os
import sys
import pickle
# from datetime import datetime
import pennylane as qml
import pennylane.numpy as np
import jax
import jax.numpy as jnp
import optax
# import matplotlib.pyplot as plt

# Adding parent directory to path for importing custom modules
parent = os.path.abspath('../src')
sys.path.insert(1, parent)
from perceptron import NativePerceptron

def parse_args():
    parser = argparse.ArgumentParser(description='Run quantum perceptron simulation.')
    parser.add_argument('--qubits', type=int, required=True, help='Number of qubits')
    parser.add_argument('--pulses', type=int, required=True, help='Number of pulses')
    parser.add_argument('--model_time', type=float, required=True, help='Evolution time of the model')
    parser.add_argument('--loss_time', type=int, required=True, help='Evolution time of the target unitary')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the .pkl file')
    return parser.parse_args()


# Configuration settings
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Setting up the quantum perceptron problem

args = parse_args()

perceptron_qubits = args.qubits
pulse_basis = args.pulses
save_path = args.save_path
t_model = args.model_time
t_loss = args.loss_time

# ts = jnp.array([1.0])
# t = 1

dev = qml.device("default.qubit.jax", wires=perceptron_qubits)

perceptron = NativePerceptron(perceptron_qubits, pulse_basis, basis='fourier', native_coupling=1)
H = perceptron.H
H_obj, H_obj_spectrum = perceptron.get_1d_ising_hamiltonian(0.1)
V = qml.matrix(qml.evolve(H_obj, t_loss))

# Defining the loss function
@jax.jit
def loss(param_vector):
    param_list = perceptron.vector_to_hamiltonian_parameters(param_vector)
    U = qml.matrix(qml.evolve(perceptron.H)(param_list, t_model))
    return qml.math.frobenius_inner_product(jnp.conjugate(U-V), U-V).real

# Initialization and optimization settings
n_epochs = 750

# Generate a random seed based on the current time
random_seed = int(time.time() * 1000)  # time in milliseconds
param_vector = perceptron.get_random_parameter_vector(random_seed)

initial_gradients = jax.grad(loss)(param_vector)

# schedule = optax.join_schedules(
#     [optax.constant_schedule(v) for v in [0.1, 0.01, 0.001]],
#     [200, 3000]
# )


# optimizer = optax.adam(learning_rate=schedule)

lr = 0.01

optimizer = optax.adam(learning_rate= lr)

opt_state = optimizer.init(param_vector)


# Storing simulation data
energies = np.zeros(n_epochs)
mean_gradients = np.zeros(n_epochs)
gradient_norms = np.zeros(n_epochs)
gradient_diffs = np.zeros(n_epochs - 1)
gradients_trajectory = []
param_trajectory = []

# Printing simulation parameters
print(f"Simulation Parameters: Qubits: {perceptron_qubits}, Pulse basis: {pulse_basis}")
print(f"Number of parameters: {param_vector.size}")
print(f"Adam Optimizer Parameters: Learning Rate {lr}, Epochs {n_epochs}")

print("Training is about to start...")

# Optimization loop
for n in range(n_epochs):
    val, grads = jax.value_and_grad(loss)(param_vector)
    updates, opt_state = optimizer.update(grads, opt_state)

    mean_gradients[n] = np.mean(np.abs(grads))
    gradient_norms[n] = jnp.linalg.norm(grads)
    energies[n] = val
    param_trajectory.append(param_vector)
    gradients_trajectory.append(grads)
    if n > 0:
        gradient_diffs[n - 1] = jnp.linalg.norm(grads - gradients_trajectory[-2])
    param_vector = optax.apply_updates(param_vector, updates)

    if n % 250 == 0:  # Adjust the frequency of printing as needed
        print(f"Epoch {n+1}/{n_epochs}; Frobenius norm: {val}")
        
        gradient_norms[n] = jnp.linalg.norm(grads)
        print(f"    Mean gradient: {mean_gradients[n]}")
        print(f"    Gradient norm: {gradient_norms[n]}")
        if n > 0:
            print(f"    Difference in gradients: {gradient_diffs[n - 1]}")

# # Calculating the Final Hessian
# final_hessian = jax.jacrev(jax.jacrev(loss))(param_vector)
# final_hessian_eigenvalues = jnp.linalg.eigvals(final_hessian)
# max_eigenvalue = jnp.max(final_hessian_eigenvalues)
# min_eigenvalue = jnp.min(final_hessian_eigenvalues)
# eigenvalue_ratio = max_eigenvalue / min_eigenvalue

# Printing final results
print(f"Training complete. Optimal Frobenius Norm Found: {energies[-1]}")
# print(f"Max eigenvalue of Hessian: {max_eigenvalue}")
# print(f"Min eigenvalue of Hessian: {min_eigenvalue}")
# print(f"Ratio of max to min eigenvalues: {eigenvalue_ratio}")

minimum_loss = np.min(energies)

print(f"Minimum Norm Achieved: {minimum_loss}")

# Saving data using pickle
simulation_data = {
    "energies": energies,
    "minimum_loss": minimum_loss,
    "mean_gradients": mean_gradients,
    "gradient_norms": gradient_norms,
    "gradient_diffs": gradient_diffs,
    "gradients_trajectory": gradients_trajectory,
    "param_trajectory": param_trajectory,
    "final_parameters": param_vector,
    "simulation_settings": {
        "qubits": perceptron_qubits,
        "pulses": pulse_basis,
        "t_model": t_model,
        "t_loss":t_loss,
        "epochs": n_epochs,
        "lr": lr
    }
}

with open(save_path, 'wb') as file:
        pickle.dump(simulation_data, file)
