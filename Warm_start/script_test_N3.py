#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:12:25 2023

@author: HP
"""
import pickle 
from perceptron import NativePerceptron
import pennylane as qml
import pennylane.numpy as np
import jax
import jax.numpy as jnp
import optax
import jaxopt
from perceptron import NativePerceptron
import time 

def prob(ExpX, beta):
    return np.exp(-beta*ExpX)
def GProb(x, y, xi):
    return (1/np.sqrt(2*np.pi*xi**2)**len(x))*np.exp(-np.sum((x-y)**2/(2*xi**2)))

Ntrials=1
N=3
P=5*N
# Setting up the quantum perceptron problem
perceptron_qubits = N
n_axis=2
pulse_basis = P
sigma=0.1
save_path = ''
n_epochs = 200


# Configuration settings
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
ts = jnp.array([1.0])
t = 1
times = jnp.linspace(0,t, pulse_basis+2)[1:-1]
dev = qml.device("default.qubit.jax", wires=perceptron_qubits)

#Setting up perceptron
perceptron = NativePerceptron(perceptron_qubits, pulse_basis, basis='gaussian', pulse_width=sigma, native_coupling=1)
H = perceptron.H

#Setting up target unitary
params_toy = np.ones(2*N*P, requires_grad=False)
params_list = perceptron.vector_to_hamiltonian_parameters(params_toy)
W = qml.matrix(qml.evolve(perceptron.H)(params_list, 1.0))

hcs = [qml.PauliX(n) for n in range(perceptron_qubits)]
hcs+= [qml.PauliY(n) for n in range(perceptron_qubits)]
# Defining the loss function
@jax.jit
def loss(param_vector):
    param_list = perceptron.vector_to_hamiltonian_parameters(param_vector)
    U = qml.matrix(qml.evolve(perceptron.H)(param_list, t))
    return qml.math.frobenius_inner_product(jnp.conjugate(U-W), U-W).real


#Setting up MCMC settings 
xi=0.1
n_epochs = 2400
n_epochs_explore=400
schedule = 0.1

beta = 0.1 #constant 

curves_warm_start = np.zeros([Ntrials, 1+n_epochs-n_epochs_explore])
curves_cold_start = np.zeros([Ntrials, 1+n_epochs-n_epochs_explore])

for trial in range(Ntrials):
    random_seed = int(time.time() * 1000)  # time in milliseconds
    params_warm = perceptron.get_random_parameter_vector(random_seed)
    optimizer_warm = optax.adam(learning_rate=schedule)
    opt_state_warm = optimizer_warm.init(params_warm)
    
    #Storing for cold start
    params_cold = params_warm.copy()
    optimizer_cold = optax.adam(learning_rate=schedule)
    opt_state_cold = optimizer_cold.init(params_cold)
    
    #PREPARING WARM START
    best_loss=loss(params_warm)
    curves_cold_start[trial,0]=best_loss
    best_params=params_warm.copy()
    
    print('Preparing warm start...')
    for i in range(n_epochs_explore):
        val, grads = jax.value_and_grad(loss)(params_warm)
        updates, opt_state_warm = optimizer_warm.update(grads, opt_state_warm)
        
        x = optax.apply_updates(params_warm, updates)
        y = x + xi*np.random.uniform(0,1,size=x.size,requires_grad=False)

        a = np.min([1.0, np.exp(-beta*(loss(y)-loss(x)))])
        u=np.random.uniform()
        
        if u<=a: 
            params_warm = y
            val=loss(params_warm)
        else:
            params_warm = x
    
        if val<best_loss: 
           best_loss=val
           best_params = params_warm
           best_opt_state = opt_state_warm
    
    print('Best loss: ', best_loss)
    
    #Restarting things
    params_warm = best_params
    
    opt_state_warm = optimizer_warm.init(params_warm)
    opt_state_warm=best_opt_state
    
    curves_warm_start[trial,0]=best_loss
    print('Optimizing both with a warm and cold starts...')
    
    #Optimization
    for i in range(n_epochs-n_epochs_explore):
        #Warm system
        val, grads = jax.value_and_grad(loss)(params_warm)
        updates, opt_state_warm = optimizer_warm.update(grads, opt_state_warm)
        #params_warm -= schedule(i+n_epochs_explore)*grads
        params_warm = optax.apply_updates(params_warm, updates)
        curves_warm_start[trial,i+1]=val
        #print('New loss for warm start: ', val)
        
        #Cold system 
        val, grads = jax.value_and_grad(loss)(params_cold)
        updates, opt_state_cold = optimizer_cold.update(grads, opt_state_cold)
        #params_cold -= schedule(i+n_epochs_explore)*grads
        params_cold = optax.apply_updates(params_cold, updates)
        curves_cold_start[trial,i+1]=val
        #print('New loss for cold start: ', val)
    
    print('Last lost with warm start: ', curves_warm_start[trial, -1])
    print('Last lost with cold start: ', curves_cold_start[trial, -1])
    