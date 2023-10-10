import pennylane as qml
import pennylane.numpy as np
import jax.numpy as jnp
from scipy.special import legendre
import jax
import optax

from time import time
from datetime import datetime
import pickle

import matplotlib.pyplot as plt


import os, sys, argparse

parent = os.path.abspath('../src')
sys.path.insert(1, parent)

from perceptron import Perceptron


# Set to float64 precision and remove jax CPU/GPU warning
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--qubits', type=int, required=True)
parser.add_argument('--field', type=float, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('-o', type=str, required=True)

parser.add_argument('--basis', type=int, required=True)
parser.add_argument('--sigma', type=float, required=True)
parser.add_argument('--k', type=int, required=True)
parser.add_argument('--n', type=int, required=True)
parser.add_argument('-outdir', type=str, required=True)




args = parser.parse_args()


def get_loss_function(perceptron, ts, H_obj, dev):

    @jax.jit
    @qml.qnode(dev, interface="jax")
    def loss(param_vector):

        # hamitlonian_params = dictionary_to_hamiltonian_parameters(params_dict)

        param_list = perceptron.vector_to_hamiltonian_parameters(param_vector)

        qml.evolve(perceptron.H)(param_list, ts)

        return qml.expval(H_obj)

    return loss


# setting up the problem
perceptron_qubits = argparse.qubits
transverse_field_coefficient = argparse.field
jax_seed = argparse.seed


fourier_basis = 5

ts = jnp.array([0.0, 1.0])

dev = qml.device("default.qubit.jax", wires = perceptron_qubits)

perceptron = Perceptron(perceptron_qubits, fourier_basis, basis='gaussian')

H =  perceptron.H

H_obj, H_obj_spectrum = perceptron.get_1d_ising_hamiltonian(transverse_field_coefficient)

e_ground_state_exact = H_obj_spectrum[0]

print(f'Ising Model Hamiltonian:\nH = {H_obj}')
print(f'Exact ground state energy: {e_ground_state_exact}')

print('###########################################################################################\n')

print(f'Starting simulation for {perceptron_qubits} qubit percetron with {transverse_field_coefficient} transverse field coefficient:')
print(f'    Training information:\n        {100} epochs with learning rate of {1}')


# getting the loss_function
loss = get_loss_function(perceptron, ts, H_obj, dev)

# getting random param_vector

param_vector = perceptron.get_random_parameter_vector(0)

print(f'Initial parameters: {param_vector}')

print(f'Initial loss: {loss(param_vector)}')

initial_gradients = jax.grad(loss)(param_vector)
print(f'Initial gradients: {initial_gradients}')

value_and_grad = jax.jit(jax.value_and_grad(loss))


n_epochs = 100
param_vector = perceptron.get_random_parameter_vector(jax_seed)

# The following block creates a constant schedule of the learning rate
# that increases from 0.1 to 0.5 after 10 epochs
# schedule0 = optax.constant_schedule(1e-1)
# schedule1 = optax.constant_schedule(5e-1)
# schedule = optax.join_schedules([schedule0, schedule1], [20])
# optimizer = optax.adam(learning_rate=schedule)
optimizer = optax.adam(learning_rate=1)
opt_state = optimizer.init(param_vector)

energy = np.zeros(n_epochs + 1)
energy[0] = loss(param_vector)
mean_gradients = np.zeros(n_epochs)
param_trajectory = []

## Compile the evaluation and gradient function and report compilation time
time0 = time()
_ = value_and_grad(param_vector)
time1 = time()

print(f"grad and val compilation time: {time1 - time0}")


## Optimization loop
for n in range(n_epochs):
    val, grads = value_and_grad(param_vector)
    updates, opt_state = optimizer.update(grads, opt_state)
    param_vector = optax.apply_updates(param_vector, updates)

    mean_gradients[n] = np.mean(np.abs(grads))
    energy[n+1] = val

    # print(f"            param: {param_vector}")

    if not n % 10:
        print(f"{n+1} / {n_epochs}; energy discrepancy: {val-e_ground_state_exact}")
        print(f"mean grad: {mean_gradients[n]}")



print('###########################################################################################\n')

print('Results: ')

print(f"    Found ground state: {energy[-1]}")



# Saving simulation results to dictionary
simulation_results_dictionary = {
    'parameters_trajectory': parameter_trajectory,
    'cost_trajectory': cost_trajectory,
}

simulation_results_dictionary['final_hessian'] = {
    'hessian_matrix': hessian_matrix,
    'determinant': determinant,
    'eigenvalues': eigenvalues,
    'eigenvectors': eigenvectors,
    'min_eigval': min_eigval,
    'max_eigval': max_eigval,
    'minmax_ratio': minmax_ratio
}




data_dictionary = {
      "qubits": perceptron_qubits,
      "epochs": 100,
      "transverse_field_coefficient": transverse_field_coefficient,
      'learning_rate': 1,
}
