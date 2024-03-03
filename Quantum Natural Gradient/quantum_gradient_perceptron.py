import sys
import os
parent = os.path.abspath('../src')
sys.path.insert(1, parent)

import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp
import time 
from multiprocessing import Pool,cpu_count
from perceptrons import NativePerceptron
import optax
import pickle
import argparse


# Configuration settings
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Run quantum perceptron simulation.')
    parser.add_argument('--qubits', type=int, required=True, help='Number of qubits')
    parser.add_argument('--pulses', type=int, required=True, help='Number of pulses')
    parser.add_argument('--lr', type =float, required=True, help='Learning Rate for Adam Optimizer')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the .pkl file')
    return parser.parse_args()


args = parse_args()

# Setting up the quantum perceptron problem

N = args.qubits
pulse_basis = args.pulses
save_path = args.save_path
lr = args.lr

# N = 4
# pulse_basis = 2*(N)
# save_path = ''


T = 1
dev = qml.device("default.qubit.jax", wires=N)
perceptron = NativePerceptron(N, pulse_basis, basis='fourier', pulse_width=0.005)

#Setting up target unitary
# H_obj, H_obj_spectrum = perceptron.get_1d_ising_hamiltonian(0.1)


H_obj = perceptron.get_1d_ising_hamiltonian(0.1)
V = qml.matrix(qml.evolve(H_obj, 1))

#Setting up Monte-Carlo Integration 
bint = 2
dt = T/bint

# Defining the loss function
@jax.jit
def loss(param_vector):
    #Calculates the loss
    param_list = perceptron.vector_to_hamiltonian_parameters(param_vector)
    U = qml.matrix(qml.evolve(perceptron.H)(param_list, T))
    return qml.math.frobenius_inner_product(jnp.conjugate(U-V), U-V).real

@jax.jit
def two_time_covariance(t1, O1, t2, O2, params_list):
    #Covariance of two operators at two distinct times. Assumes t1>t2
    psi0 = jnp.eye(2**(perceptron.n_qubits), dtype=complex)[0] # |0>
    U_t1 = qml.matrix(qml.evolve(perceptron.H)(params_list, t1))
    U_t2 = qml.matrix(qml.evolve(perceptron.H)(params_list, t2))
    psi_t1 = U_t1 @ psi0
    psi_t2 = U_t2 @ psi0
    
    u_partial= qml.matrix(qml.evolve(perceptron.H)(params_list, t=[t2,t1]))

    # print("psi_t1 shape:", psi_t1.shape)
    # print("O1 shape:", O1.shape)
    # print("u_partial shape:", u_partial.shape)
    # print("O2 shape:", O2.shape)

    return psi_t1.conj().T @ O1 @ u_partial@ O2 @ u_partial.conj().T @ psi_t1 - (psi_t1.conj() @ O1 @psi_t1)*(psi_t2.conj()@ O2 @psi_t2)
    
@jax.jit
def covariance(t, O1, O2, params_list):
    #Covariance for the case that the two times coincide (t1=t2)
    psi0 = jnp.eye(2**(perceptron.n_qubits), dtype=complex)[0]
    U_t = qml.matrix(qml.evolve(perceptron.H)(params_list, t))    
    psi_t = U_t @ psi0
    return psi_t.conj().T @ O1@O2 @ psi_t-(psi_t.conj()@ O1 @psi_t)*(psi_t.conj()@ O2 @psi_t)

def diagonal_MCI(index,params_list):
    #This function calculates a single component of the QNG on the diagonal 

    #Index is 0,...,2*(pulse_basis)*(N+1)
    axis  = index//((N)*(pulse_basis))
    site  = index//(2*(pulse_basis))
    basis = index//(2*(N))
    
    if   axis==0: O= qml.matrix(qml.PauliX(site),wire_order=range(N))
    elif axis==1: O= qml.matrix(qml.PauliY(site),wire_order=range(N))

    def g(t):
        return np.cos(2*np.pi*basis*t/pulse_basis)
    
    s=0.0
    for b in range(bint):
        ti = np.random.uniform(low=0,high=T)
        tj = np.random.uniform(low=0,high=T)
        if ti>tj:    s+= T **2 *(g(ti)*g(tj)*two_time_covariance(ti, O, tj, O, params_list))
        elif ti==tj: s+= T **2 *(g(ti)**2*covariance(ti,O, O, params_list))
        else:        s+= T **2 *(g(ti)*g(tj)*two_time_covariance(tj, O, ti, O, params_list))
    
    return np.real(s)





def get_diagonal_metric_tensor(param_vector):


    param_list = perceptron.vector_to_hamiltonian_parameters(param_vector)


    dimension = N*pulse_basis
    
    metric_tensor = jnp.zeros((dimension, dimension))

    for ii in range(dimension):
        
        metric_tensor = metric_tensor.at[ii,ii].set(diagonal_MCI(ii, param_list))


    return metric_tensor



def get_qng_grad(param_vector, grad):

    param_list = perceptron.vector_to_hamiltonian_parameters(param_vector)

    dimension = param_vector.shape[0]

    qng = jnp.zeros(dimension)
    
    for ii in range(dimension):
        
        qng = qng.at[ii].set(diagonal_MCI(ii, param_list) * grad[ii])

    return qng





param_vector = perceptron.get_random_parameter_vector(0)



print(f'Initial parameters: {param_vector}')

value_and_grad = jax.jit(jax.value_and_grad(loss))

metric_tensor = get_diagonal_metric_tensor(param_vector)

val, grad = value_and_grad(param_vector)

qng_grad = get_qng_grad(param_vector, grad)


print(f'Initial loss: {val}')

print(f'Initial gradients (no QNG): {grad}')

print(f'Initial gradients (with QNG): {qng_grad}')


#################################################################################################################################

from datetime import datetime

n_epochs = 4
param_vector = perceptron.get_random_parameter_vector(808)


optimizer = optax.adam(learning_rate=lr)
opt_state = optimizer.init(param_vector)

energies = []
mean_gradients = []
gradients_norms = []

gradients_trajectory = []
qng_gradients_trajectory = []
param_trajectory = []


## Optimization loop
for n in range(n_epochs):

    val, grads = value_and_grad(param_vector)

    qng_grads = get_qng_grad(param_vector, grads)

    updates, opt_state = optimizer.update(qng_grads, opt_state)

    mean_gradients.append(jnp.mean(jnp.abs(grads)))
    energies.append(val)
    gradients_norms.append(np.linalg.norm(qng_grad))
    param_trajectory.append(param_vector)
    gradients_trajectory.append(grads)
    qng_gradients_trajectory.append(qng_grads)

    param_vector = optax.apply_updates(param_vector, updates)

    # print(val)
    # print(jnp.linalg.norm(qng_grads))
    # mean_gradients[n] = np.mean(np.abs(grads))
    # energy[n+1] = val

    # print(f"            param: {param_vector}")

print(f"Found ground state: {energies[-1]}")


results_dictionary = {
    'n_qubits': N,
    'pulse_basis': pulse_basis,
    'gradients_trajectory': gradients_trajectory,
    'energies': energies,
    'qng_gradients_trajectoy': qng_gradients_trajectory,
    'qng_gradients_norms': gradients_norms
}


print(results_dictionary)


with open(save_path, 'wb') as file:
        pickle.dump(results_dictionary, file)
