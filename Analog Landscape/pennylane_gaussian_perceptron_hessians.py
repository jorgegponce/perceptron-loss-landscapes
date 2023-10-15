import pennylane as qml
import pennylane.numpy as np
import jax.numpy as jnp
from scipy.special import legendre
import jax
import optax

from time import time
from datetime import datetime
import pickle

# import matplotlib.pyplot as plt


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
# parser.add_argument('--k', type=int, required=True)
# parser.add_argument('--n', type=int, required=True)
# parser.add_argument('-outdir', type=str, required=True)




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
perceptron_qubits = args.qubits
transverse_field_coefficient = args.field
jax_seed = args.seed
n_pulse_basis = args.basis
pulse_width = args.sigma
out_file_path = args.o

ts = jnp.array([0.0, 1.0])

dev = qml.device("default.qubit.jax", wires = perceptron_qubits)

perceptron = Perceptron(perceptron_qubits, n_pulse_basis, basis='gaussian', pulse_width=pulse_width)

H =  perceptron.H

H_obj, H_obj_spectrum = perceptron.get_1d_ising_hamiltonian(transverse_field_coefficient)

e_ground_state_exact = H_obj_spectrum[0]

print(f'Ising Model Hamiltonian:\nH = {H_obj}')
print(f'Exact ground state energy: {e_ground_state_exact}')

print('###########################################################################################\n')

print(f'Starting simulation for {perceptron_qubits} qubit percetron with {transverse_field_coefficient} transverse field coefficient:')

# getting the loss_function
loss = get_loss_function(perceptron, ts, H_obj, dev)

# getting random param_vector

param_vector = perceptron.get_random_parameter_vector(0)

print(f'Initial parameters: {param_vector}')

print(f'Initial loss: {loss(param_vector)}')

initial_gradients = jax.grad(loss)(param_vector)
print(f'Initial gradients: {initial_gradients}')

value_and_grad = jax.jit(jax.value_and_grad(loss))


n_epochs = 400
param_vector = perceptron.get_random_parameter_vector(jax_seed)

# The following block creates a constant schedule of the learning rate
# that increases from 0.1 to 0.5 after 10 epochs
schedule0 = optax.constant_schedule(1)
schedule1 = optax.constant_schedule(0.2)
schedule = optax.join_schedules([schedule0, schedule1], [50])
optimizer = optax.adam(learning_rate=schedule)
# optimizer = optax.adam(learning_rate=1)
opt_state = optimizer.init(param_vector)

energies = np.zeros(n_epochs )
# energy[0] = loss(param_vector)
mean_gradients = np.zeros(n_epochs)

gradients_trajectory = []
param_trajectory = []

## Compile the evaluation and gradient function and report compilation time
time0 = time()
_ = value_and_grad(param_vector)
time1 = time()

# print(f"grad and val compilation time: {time1 - time0}")


## Optimization loop
for n in range(n_epochs):
    val, grads = value_and_grad(param_vector)
    updates, opt_state = optimizer.update(grads, opt_state)

    mean_gradients[n] = np.mean(np.abs(grads))
    energies[n] = val
    param_trajectory.append(param_vector)
    gradients_trajectory.append(grads)

    param_vector = optax.apply_updates(param_vector, updates)

    # mean_gradients[n] = np.mean(np.abs(grads))
    # energy[n+1] = val

    # print(f"            param: {param_vector}")

    if not n % 10:
        print(f"{n+1} / {n_epochs}; energy discrepancy: {val-e_ground_state_exact}")
        print(f"mean grad: {mean_gradients[n]}")
        print(f'gradient norm: {jnp.linalg.norm(grads)}')
        if n>=2:
            print(f'difference of gradients: {jnp.linalg.norm(grads-gradients_trajectory[-2])}')



print('###########################################################################################\n')

print('Results: ')

print(f"    Found ground state: {energies[-1]}")

# print(f' Test: {value_and_grad(param_trajectory[-1])}')

gradients_norms = [jnp.linalg.norm(gradient) for gradient in gradients_trajectory]
differences_of_gradients_norms = [jnp.linalg.norm(gradients_trajectory[j] - gradients_trajectory[j-1]) for j in range(1,len(gradients_trajectory))]


# Calculating the Hessian at the final point

final_hessian = jax.jacrev(jax.jacrev(loss))(param_trajectory[-1])
final_hessian_eigenvalues = jnp.linalg.eigvals(final_hessian)
print(final_hessian_eigenvalues)
min_eigval = jnp.min(final_hessian_eigenvalues)
max_eigval = jnp.max(final_hessian_eigenvalues)
minmax_ratio = min_eigval/max_eigval

print(f'Min/Max ratio: {minmax_ratio}')

# Saving simulation results to dictionary
simulation_results_dictionary = {
    'parameters_trajectory': param_trajectory,
    'energy_trajectory': energies,
    'gradient_trajectory': gradients_trajectory,
    'mean_gradients': mean_gradients,
    'gradients_norms': gradients_norms,
    'differences_of_gradients_norms': differences_of_gradients_norms
}

simulation_results_dictionary['final_hessian'] = {
    'hessian_matrix': final_hessian,
    'eigenvalues': final_hessian_eigenvalues,
    'min_eigval': min_eigval,
    'max_eigval': max_eigval,
    'minmax_ratio': minmax_ratio
}

data_dictionary = {
      'qubits': perceptron_qubits,
      'epochs': 400,
      'transverse_field_coefficient': transverse_field_coefficient,
      'learning_rates': (1, 0.2),
      'hamiltonian_spectrum': H_obj_spectrum,
      'exact_gs_energy': e_ground_state_exact,
      'simulation_results': simulation_results_dictionary
}


import pickle

# save dictionary to person_data.pkl file
with open(out_file_path, 'wb') as fp:
    pickle.dump(data_dictionary, fp)
    print('dictionary saved successfully to file')


with open(out_file_path, 'rb') as fp:
    loaded_dict = pickle.load(fp)


print(loaded_dict)
