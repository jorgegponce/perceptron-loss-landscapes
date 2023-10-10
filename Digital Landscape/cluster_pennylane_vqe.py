import pennylane as qml
from pennylane import numpy as np
import argparse

# import torch

# from torch.optim import Adam

from time import time
import pickle



parser = argparse.ArgumentParser()
parser.add_argument('--qubits', type=int, required=True)
parser.add_argument('--field', type=float, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('-o', type=str, required=True)
args = parser.parse_args()

n_qubits = args.qubits
transverse_field_coefficient = args.field
lr = args.lr
epochs = args.epochs
filepath = args.o

depth = n_qubits
dev1 = qml.device("default.qubit", wires=n_qubits)


# Constructing the 1D Ising Hamiltonian
def get_ising_hamiltonian(n_qubits, transverse_field_coefficient):
    
    
    obs = [qml.PauliZ(i) @ qml.PauliZ(i+1) for i in range((n_qubits-1))]
    coeffs = [1.0 for i in range((n_qubits-1))]
    z_term = qml.Hamiltonian(coeffs, obs)

    obs = [qml.PauliX(i) for i in range((n_qubits-1))]
    coeffs = [transverse_field_coefficient for i in range((n_qubits-1))]
    transverse_field_term = qml.Hamiltonian(coeffs, obs)

    ising_hamiltonian = transverse_field_term + z_term

    # if n_qubits < 5:

    #     n_eigenvalues = 2**(n_qubits-1)

    # else:
    #     n_eigenvalues = 30

    # energies = qml.eigvals(ising_hamiltonian, k=n_eigenvalues, which='SR')

    return ising_hamiltonian #, energies



def perceptron_circuit(param_list, n_qubits, depth):

    gate_offset = 1 * n_qubits # number of parameters per operator times number of operators
    
    for d in range(depth):

        depth_offset = d*gate_offset*4
        # print(depth_offset)

        # print("new layer with depth offset ", depth_offset)

        for i in range(n_qubits):
            theta = param_list[i + gate_offset*0 + depth_offset]
            # print("RX layer ", param_list[i + gate_offset*0 + depth_offset])
            # gate = RX(i, param_list[i + gate_offset*0 + depth_offset])
            qml.RX(theta, wires=i)

        for i in range(n_qubits):
            theta = param_list[i + gate_offset*1 + depth_offset]
            # print("RZ layer ", param_list[i + gate_offset*1 + depth_offset])
            # gate = RZ(i, param_list[i + gate_offset*1 + depth_offset])
            qml.RZ(theta, wires = i)


        for i in range(n_qubits-1):
            qml.CZ(wires = [i, n_qubits-1])

        for i in range(n_qubits):
            theta = param_list[i + gate_offset*2 + depth_offset]
            # print("RX layer ", param_list[i + gate_offset*2 + depth_offset])
            # gate = RX(i, param_list[i + gate_offset*2 + depth_offset])
            qml.RX(theta, wires = i)

        for i in range(n_qubits):
            theta = param_list[i+ gate_offset*3 + depth_offset]
            # print("RZ layer ", param_list[i + gate_offset*3 + depth_offset])
            # gate = RZ(i, param_list[i+ gate_offset*3 + depth_offset])
            qml.RZ(theta, wires = i)




def gradient_descent_optimizer(initial_parameters, 
                               loss_function,
                               n_iters=80,
                               learning_rate=0.2):
    opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
    parameter_trajectory = [initial_parameters, ]
    losses = [loss_function(initial_parameters), ]
    params = initial_parameters

    for i in range(n_iters):
        params, prev_cost = opt.step_and_cost(loss_function, params)
        cost = loss_function(params) # this is squandering executions ...
        parameter_trajectory += [params, ]
        losses += [cost, ]
        print(f'Epoch number: {i+1} | loss : {cost}')

    return parameter_trajectory, losses


@qml.qnode(dev1, interface='autograd')
def loss_function(param_list):

    perceptron_circuit(param_list, n_qubits, depth)

    ising_hamiltonian = get_ising_hamiltonian(n_qubits, transverse_field_coefficient)

    return qml.expval(ising_hamiltonian)



print(f'Running experiment for {n_qubits} qubits with h = {transverse_field_coefficient} | Results saved at {filepath}')

gates_per_layer = n_qubits
param_dim = gates_per_layer*4*depth
initial_parameters = np.random.random(param_dim)*1e-1


hamiltonian = get_ising_hamiltonian(n_qubits, transverse_field_coefficient)
hamiltonian_spectrum = qml.eigvals(hamiltonian, which='SR')
gs_energy = np.min(hamiltonian_spectrum)

print('###########################################################################################\n')

print(f'Starting simulation for {n_qubits} qubit percetron with {transverse_field_coefficient} transverse field coefficient:')
print(f'    Training information:\n        {epochs} epochs with learning rate of {lr}')
print(f'    Perceptron Circuit:\n')


print(qml.draw(perceptron_circuit)(initial_parameters, n_qubits, depth))

print('\n\n')

st = time()
parameter_trajectory, losses = gradient_descent_optimizer(initial_parameters, loss_function, n_iters=epochs, learning_rate=lr)
et = time()


final_parameters = parameter_trajectory[-1]

print('###########################################################################################\n')

print('Results: ')

print(f'    Achieved ground state:  {losses[-1]}')

print(f'    True Ground State: {gs_energy}')

print(f'\n    Training time: {et-st} seconds')

print('###########################################################################################\n')
print('Final loss (sanity check): ')

print(f'{loss_function(final_parameters)}\n')

print('###########################################################################################\n')
print('Final gradient: ')


st = time()
dloss = qml.grad(loss_function)
final_gradient = dloss(final_parameters)
et = time()

# print(f'    Final loss: {final_loss}')
print(f'    Final gradient: {final_gradient}')
print(f'        Norm: {np.linalg.norm(final_gradient)}')

print(f'\n    Execution time: {et-st} seconds')

print('###########################################################################################\n')

print(f"(AUTOGRAD) Calculating final Hessian")


st = time()
ddloss = qml.jacobian(dloss)

final_hessian = ddloss(final_parameters)

(eigenvalues, eigenvectors) = np.linalg.eigh(final_hessian)

min_eigval = eigenvalues[0]
max_eigval = eigenvalues[-1]
minmax_ratio = min_eigval/max_eigval
determinant = np.linalg.det(final_hessian)

et = time()

print(f'Hessian information: ')
print(f'    Dimensions: {final_hessian.shape}')
print(f'        Eigenvalue ratio: {minmax_ratio} | Minimum eigenvalue: {min_eigval} | Maximum eigenvalue: {max_eigval}')
print(f'        Determinant: {determinant}')
print(f'        Parameter vector: {final_parameters}\n')
print(f'\n        Execution time:   {et - st} seconds')



# Saving results to a dictionary:


problem = {
        "qubits": n_qubits,
        "transverse_field_coefficient": transverse_field_coefficient,
        "learning_rate": lr,
        "epochs": epochs,
        'loss_hamiltonian': hamiltonian,
        'loss_hamiltonian_spectrum': hamiltonian_spectrum,
        'simulation_results': {
            'parameters_trajectory': parameter_trajectory,
            'cost_trajectory': losses,
            'final_gradient': final_gradient,
            'final_hessian': {
                'hessian_matrix': final_hessian,
                'determinant': determinant,
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'min_eigval': min_eigval,
                'max_eigval': max_eigval,
                'minmax_ratio': minmax_ratio
        },
    },
}


pickle_file = filepath


with open(pickle_file, 'wb') as handle:

    # for problem in vqe_problems:
    #     problem['loss_hamiltonian'] = None

    pickle.dump(problem, handle)


# test to see that the hessian eigenopbject is safely stored

# with open(pickle_file, 'rb') as handle:

#     # for problem in vqe_problems:
#     #     problem['loss_hamiltonian'] = None

#     test_dictionary = pickle.load(handle)

# # print(test_dictionary['simulation_results']['final_hessian'])
# print(test_dictionary['loss_hamiltonian'])
# print(test_dictionary['loss_hamiltonian_spectrum'])
