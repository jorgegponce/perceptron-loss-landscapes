#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:46:04 2023

@author: HP
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 10:06:21 2023

@author: HP
"""
import matplotlib.pyplot as plt
from perceptron import NativePerceptron
import pennylane as qml
import pennylane.numpy as np
from pennylane import qchem 
import jax
import jax.numpy as jnp
import optax
import time 

symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
print("Number of qubits = ", qubits)
print("The Hamiltonian is ", H)
E0 = qml.eigvals(H, k=6, which='SR')[0]


# Setting up the quantum perceptron problem
perceptron_qubits = qubits+1
n_axis=2
pulse_basis = 2*perceptron_qubits
sigma=0.01
save_path = ''
n_epochs = 200

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
ts = jnp.array([1.0])
t = 1
times = jnp.linspace(0,t, pulse_basis+2)[1:-1]
dev = qml.device("default.qubit.jax", wires=perceptron_qubits)


#Setting up perceptron
perceptron = NativePerceptron(perceptron_qubits, pulse_basis, basis='fourier', pulse_width=sigma, native_coupling=1)


#Defining the loss:
@jax.jit
@qml.qnode(dev, interface="jax")
def loss(param_vector):
    param_list = perceptron.vector_to_hamiltonian_parameters(param_vector)
    qml.evolve(perceptron.H)(param_list, t)
    return qml.expval(H)

value_and_grad = jax.jit(jax.value_and_grad(loss))

n_epochs = 1000
n_explore = 200 
Nreps=5
schedule = 0.1
xi=0.05

beta = 0.1 #constant 

curves_explore = np.zeros([Nreps, n_explore])
curves_warm_start = np.zeros([Nreps,1+n_epochs])
curves_cold_start = np.zeros([Nreps, 1+n_epochs])

for rep in range(Nreps):
    
    random_seed = int(time.time() * 1000)  # time in milliseconds
    params = perceptron.get_random_parameter_vector(random_seed)
    Init_loss =loss(params)
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params)
    curves_warm_start[rep, 0] = Init_loss
    curves_cold_start[rep, 0] = Init_loss

    #Starting for warm start
    params_warm = params
    optimizer_warm = optax.adam(learning_rate=schedule)
    opt_state_warm = optimizer_warm.init(params)

    #Storing for cold start
    params_cold = params_warm.copy()
    optimizer_cold = optax.adam(learning_rate=schedule)
    opt_state_cold = optimizer_cold.init(params_cold)

    #PREPARING WARM START
    best_loss=Init_loss
    best_params=params_warm.copy()
    
    print('Random parameters give initial accuracy: ', abs((Init_loss-E0)/E0))
    
    print('Preparing warm start...')
    for i in range(n_explore):
        val, grads = value_and_grad(params_warm)
        updates, opt_state_warm = optimizer_warm.update(grads, opt_state_warm)
        
        x = optax.apply_updates(params_warm, updates)
        y = x + xi*np.random.uniform(0,1,size=x.size,requires_grad=False)
    
        a = np.min([1.0, np.exp(-beta*(loss(y)-loss(x)))])
        u=np.random.uniform()
        
        if u<=a: 
            params_warm = y
        else:
            params_warm = x
        
        new_val = loss(params_warm)
    
        if new_val<best_loss: 
           best_loss=new_val
           best_params = params_warm
           best_opt_state = opt_state_warm
        
        curves_explore[rep, i] = val

    #Now, we optimize over warm start or just using a cold start 
    print("Best accuracy achieved during exploration: ", abs((best_loss-E0)/E0))
    curves_warm_start[rep, 0] = best_loss
    
    #Restarting things
    params_warm = best_params
    
    opt_state_warm = optimizer_warm.init(params_warm)
    opt_state_warm=best_opt_state


    print('Optimizing both with a warm and cold starts...')
    
    #Optimization
    for i in range(n_epochs):
        #Warm system
        val, grads = jax.value_and_grad(loss)(params_warm)
        updates, opt_state_warm = optimizer_warm.update(grads, opt_state_warm)
        #params_warm -= schedule(i+n_epochs_explore)*grads
        params_warm = optax.apply_updates(params_warm, updates)
        curves_warm_start[rep, i+1]=val
        #print('New loss for warm start: ', val)
        
        #Cold system 
        val, grads = jax.value_and_grad(loss)(params_cold)
        updates, opt_state_cold = optimizer_cold.update(grads, opt_state_cold)
        #params_cold -= schedule(i+n_epochs_explore)*grads
        params_cold = optax.apply_updates(params_cold, updates)
        curves_cold_start[rep, i+1]=val
        #print('New loss for cold start: ', val)
    
    print('Last lost with warm start: ', curves_warm_start[rep,-1])
    print('Last lost with cold start: ', curves_cold_start[rep, -1])

fig, axes = plt.subplots(nrows=Nreps, ncols=2)
start=100
for r in range(Nreps):
    axes[r,0].plot(range(start,n_epochs+1), abs((curves_cold_start[r,start:]-E0)/E0), linewidth=3)
    axes[r,0].plot(range(start,n_epochs+1), abs((curves_warm_start[r,start:]-E0)/E0), linewidth=3)
    axes[r,0].set_yscale('log')
    axes[r,1].plot(range(n_explore), curves_explore[r], linewidth=3)

fig.suptitle('Beta='+str(beta))
fig.tight_layout()
plt.show()

