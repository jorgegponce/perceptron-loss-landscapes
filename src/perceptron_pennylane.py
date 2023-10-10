import pennylane as qml
from pennylane import numpy as np

class Perceptron():

    # dev1 = qml.device("default.qubit", wires=4)


    # def __init__(self, n_qubits, control_fields, loss_hamiltonian_list, time=1.0, n_basis = 5, basis='Fourier', initial_psi=None, js=None, vs=None, constrained=False):

    def __init__(self, n_qubits, time=1.0, n_basis = 5, basis='Fourier', initial_psi=None, js=None, vs=None, constrained=False):

        self.n_qubits = n_qubits
        # self.native_hamiltonian_graph = native_hamiltonian_graph
        # self.control_fields=control_fields
        # self.loss_hamiltonian_list=loss_hamiltonian_list
        self.time=time
        self.n_basis=n_basis
        self.basis=basis
        # self.method=method
        self.constrained = constrained
        
        # if js is None: self.js = np.random.rand(len(self.native_hamiltonian_graph))
        if js is None: self.js = np.random.rand(n_qubits-1, requires_grad = True)
        else: self.js = js
        
        # if vs is None: self.vs =  np.random.rand([len(self.control_fields), n_basis])
        if vs is None: self.vs =  np.random.rand(2*n_qubits, n_basis, requires_grad = True)
        else: self.vs = vs
        
        if initial_psi is None:
            self.initial_psi = np.zeros(2**self.n_qubits)
            self.initial_psi[0] = 1.0
        else: self.initial_psi = initial_psi
        
        # self.sx = torch.tensor([[0j,1.],[1.,0j]])
        # self.sy = torch.tensor([[0.,-1j],[1j,0.]])
        # self.sz = torch.tensor([[1.,0j],[0,-1.]])
        # self.I2 = torch.tensor([[1.0, 0j],[0,1.]])
        

        self.perceptron_hamiltonian = self.get_system_hamiltonian()



    def get_system_hamiltonian(self):

        # building the native hamiltonian
        self.native_hamiltonian_graph = [qml.PauliZ(i) @ qml.PauliZ(self.n_qubits-1) for i in range((self.n_qubits-1))]
        self.native_hamiltonian = qml.Hamiltonian(self.js, self.native_hamiltonian_graph)

        # building the control hamiltonian

        self.control_hamiltonian_operators = [qml.PauliY(i) for i in range(self.n_qubits)]
        self.control_hamiltonian_operators += [qml.PauliX(i) for i in range(self.n_qubits)]

        def system_hamiltonian(t):
            # for i in range(2*n_qubits):
            #     control_hamiltonian_coefficients.append(self.generate_p(i))

            control_hamiltonian_coefficients = [self.generate_p(i)(t) for i in range(2*n_qubits)]

            control_hamiltonian = qml.Hamiltonian(control_hamiltonian_coefficients, self.control_hamiltonian_operators)

            return control_hamiltonian + self.native_hamiltonian

        return system_hamiltonian



    def generate_p(self, i):
        """Generate the function p_i(t) for H_i
        Args:
            i: index of the H_i.
            coefficient of shape [nqubits, 2] 
        Returns:
            p: function p_i(t).
        """

        def p(t):
            if self.basis == 'Fourier':
                u = np.dot(self.vs[i,:],np.cos(2*np.pi*(t/self.time)*np.arange(self.n_basis)))
            elif self.basis == 'poly':
                u = np.dot(self.vs[i,:],(t/self.time)**np.arange(self.n_basis))
            # elif self.basis == 'Legendre':
            #     u = torch.dot(self.vs[i,:],lp(t/self.time, torch.arange(self.n_basis)))

            return u
        
        return p
        

    def setup_time_evolution_circuit(self, t, dt, trotter_number=1):

        times = np.arange(0, t, dt)

        for time in times:

            qml.ApproxTimeEvolution(self.perceptron_hamiltonian(time) , dt, trotter_number)
        

    def loss(self, t, dt, trotter_number, loss_hamiltonian):

        self.forward(t, dt, trotter_number)

        return qml.expval(loss_hamiltonian)





def tranverse_field_ising_hamiltonian(n_qubits, transverse_field_coefficient):
    pass


def loss_function(js, vs):
    pass




n_qubits = 4

dev1 = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev1, interface="autograd")
def perceptron_time_evolution():

    perceptron = Perceptron(n_qubits)

    perceptron.setup_time_evolution_circuit(1, 0.01, 1)

    return qml.state()

dev1 = qml.device("default.qubit", wires=n_qubits)

perceptron = Perceptron(n_qubits)

print(perceptron.native_hamiltonian)

print(perceptron.perceptron_hamiltonian(0))

print(qml.draw(perceptron_time_evolution)())
