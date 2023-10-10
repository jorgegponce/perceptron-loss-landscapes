import pennylane as qml
import pennylane.numpy as np
import jax.numpy as jnp
from scipy.special import legendre
from scipy.integrate import quad
import jax
import optax

from tqdm import tqdm


from time import time
import pickle

import matplotlib.pyplot as plt

import os, sys, argparse

parent = os.path.abspath('../src/')
sys.path.insert(1, parent)

from perceptron import Perceptron


# Set to float64 precision and remove jax CPU/GPU warning
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


parser = argparse.ArgumentParser()

parser.add_argument('--qubits', type=int, required=True)
parser.add_argument('--field', type=float, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--basis', type=int, required=True)
parser.add_argument('--sigma', type=float, required=True)
parser.add_argument('--k', type=int, required=True)
parser.add_argument('--n', type=int, required=True)
parser.add_argument('-outdir', type=str, required=True)

args = parser.parse_args()


# setting up the problem
perceptron_qubits = args.qubits
transverse_field_coefficient = args.field
random_seed = args.seed
pulse_basis = args.basis
pulse_width = args.sigma
k = args.k
n = args.n
outdir = args.outdir


# Checking whether directory exists
if not os.path.exists(outdir):
    os.makedirs(outdir)
    print('Succesfully created directory!')

else:
    print('Directory already exists')


filename = f'{perceptron_qubits}_qubits_{transverse_field_coefficient}_field-{pulse_width}_sigma_{k}_{n}.data'

# perceptron_qubits = 2

dev = qml.device("default.qubit.jax", wires = perceptron_qubits)

# transverse_field_coefficient = args.field
# pulse_basis = 2
# n_steps = 50
# lr = args.lr
# random_seed = 7

t0 = 0.0
tf = 1.0

perceptron = Perceptron(perceptron_qubits, pulse_basis, basis='gaussian', pulse_width=pulse_width)

H =  perceptron.H

param_vector = perceptron.get_random_parameter_vector(random_seed)
param_list = perceptron.vector_to_hamiltonian_parameters(param_vector)


# print(param_vector)

# print(param_list)


def integrand_circuit_representation(k, t, t0, tf):

    if t != t0:
        qml.evolve(H)(param_list, [t0, t], hmax = 1e-3)

    perceptron.H_control_operators[k]

    if t!= tf:
        qml.evolve(H)(param_list, [t, tf], hmax = 1e-3)


def local_surjectivity(k, n, t0, tf, n_steps):

    approx_times = jnp.linspace(t0, tf, n_steps)

    dt = approx_times[1]

    dUdtheta = jnp.zeros((2**perceptron_qubits, 2**perceptron_qubits))

    # g_n = lambda t: jnp.cos(2*jnp.pi*(t)*n)
    # g_n = lambda t: t**n
    # g_n = lambda t: 1

    g_n = perceptron.field_basis_functions[n]
    

    for t in tqdm(approx_times):
        # print(t)
        # print(t)
        integrand_matrix = qml.matrix(integrand_circuit_representation)(k, t, t0, tf)
        # print(integrand_matrix)

        dUdtheta += g_n(t) * integrand_matrix * dt

    return dUdtheta


total_control_fields = 2*perceptron_qubits

n_steps = int(1e3)
# n_steps = 2
# X = [2]


# for n_steps in X:

#     determinant_matrix = np.zeros((total_control_fields, pulse_basis))
#     # eigenvalues_matrix = np.empty((total_control_fields, pulse_basis), dtype=object)

#     for k in range(total_control_fields):
#         print(f'Field: {k+1}')
#         for n in range(pulse_basis):

#             print(f'Pulse expansion terms: {n+1}')
#             derivative = local_surjectivity(total_control_fields-1-k,n, t0, tf, n_steps)
#             eigenvalues_array = np.linalg.det(derivative)
#             determinant_matrix[k,n] = eigenvalues_array
#             print(eigenvalues_array)

#             print(derivative)


#     determinant_dictionary[f'{n_steps}'] = determinant_matrix


print(f'Field: {k+1}')
print(f'Pulse expansion term: {n+1}')
derivative_matrix = local_surjectivity(k,n, t0, tf, n_steps)
determinant = np.linalg.det(derivative_matrix)

print(determinant)

data_dictionary = {
    'n_steps': n_steps,
    'dertivative': derivative_matrix,
    'determinant': determinant
}


filepath = f'{outdir}/{filename}'

print(f"Saving dictionary as: {filepath}")



import pickle


# save dictionary to person_data.pkl file
with open(filepath, 'wb') as fp:
    pickle.dump(data_dictionary, fp)
    print('dictionary saved successfully to file')


with open(filepath, 'rb') as fp:
    loaded_dict = pickle.load(fp)


print(loaded_dict)
