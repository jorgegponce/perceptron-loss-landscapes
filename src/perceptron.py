import pennylane as qml
import pennylane.numpy as np
import jax.numpy as jnp
from jax.scipy.special import lpmn_values
import jax

class Perceptron():

    def __init__(self, n_qubits, n_basis, timespan = 1,  basis='fourier', pulse_width = 0.5e-2):
    
        self.n_qubits = n_qubits
        self.n_basis = n_basis
        self.basis = basis
        self.pulse_width = pulse_width
        self.timespan = timespan

        # self.field_basis_functions = []

        self.create_parametrized_hamiltonian()



    def create_parametrized_hamiltonian(self):

        self.create_native_hamiltonian()
        self.create_control_hamiltonian()
        
        self.H = self. H_native + self.H_control 
        

    def create_native_hamiltonian(self):

        self.native_fields = [self.get_constant_field() for i in range(self.n_qubits-1)]
        self.H_native_operators = [qml.PauliZ(i) @ qml.PauliZ(self.n_qubits-1) for i in range(self.n_qubits-1)]

        self.H_native = qml.dot(self.native_fields, self.H_native_operators)


    def create_control_hamiltonian(self):

        if self.basis == 'fourier':
            self.control_fields_x = [self.get_fourier_field() for i in range(self.n_qubits)]
            self.control_fields_y = [self.get_fourier_field() for i in range(self.n_qubits)]

        if self.basis == 'legendre':
            self.control_fields_x = [self.get_legendre_field() for i in range(self.n_qubits)]
            self.control_fields_y = [self.get_legendre_field() for i in range(self.n_qubits)]

        if self.basis == 'chebyshev':
            self.control_fields_x = [self.get_chebyshev_field() for i in range(self.n_qubits)]
            self.control_fields_y = [self.get_chebyshev_field() for i in range(self.n_qubits)]

        if self.basis == 'pwc':
            self.control_fields_x = [self.get_piecewise_constant_field() for i in range(self.n_qubits)]
            self.control_fields_y = [self.get_piecewise_constant_field() for i in range(self.n_qubits)]

        if self.basis == 'delta':
            
            self.field_basis_functions = [jax.jit(lambda t : Perceptron.delta_function(i, t)) for i in jnp.linspace(0,self.timespan,self.n_basis+2)[1:-1]]

            self.control_fields_x = [self.get_delta_field() for i in range(self.n_qubits)]
            self.control_fields_y = [self.get_delta_field() for i in range(self.n_qubits)]

        if self.basis == 'gaussian':

            self.field_basis_functions = [jax.jit(lambda t : Perceptron.gaussian(i, t, self.pulse_width)) for i in jnp.linspace(0,self.timespan,self.n_basis+2)[1:-1]]

            self.control_fields_x = [self.get_gaussian_field() for i in range(self.n_qubits)]
            self.control_fields_y = [self.get_gaussian_field() for i in range(self.n_qubits)]


        self.H_control_operators_x = [qml.PauliX(i) for i in range(self.n_qubits)]
        self.H_control_operators_y = [qml.PauliY(i) for i in range(self.n_qubits)]

        self.control_fields = self.control_fields_x + self.control_fields_y
        self.H_control_operators = self.H_control_operators_x + self.H_control_operators_y

        self.H_control = qml.dot(self.control_fields, self.H_control_operators)


    def get_constant_field(self):

        def constant_field(p,t):
            # p is the trainable parameter
            # t is the time
            return p

        return constant_field


    def get_fourier_field(self):
        
        def fourier_field(p, t):
            # p are the trainable parameters
            # t is the time
            return jnp.dot(p, jnp.cos(2*jnp.pi*(t)*jnp.arange(self.n_basis))) # QUESTION: WHY ARE WE MULTIPLYING BY 2?
        
        return fourier_field
    

    def get_legendre_field(self):
        
        def legendre_field(p, t):
            # p are the trainable parameters
            # t is the time

            # degrees = jnp.arange(self.n_basis)

            # legendre_polynomials, _ = jax.vmap(lambda d: lpmn_values(self.n_basis - 1, d, t, is_normalized = False))(degrees)
    
            legendre_polynomials = [Perceptron.legendre_polynomial(i, t) for i in range(self.n_basis)]

            legendre_polynomials = jnp.array(legendre_polynomials)

            return jnp.dot(p, legendre_polynomials)
        
        return legendre_field
    

    def get_chebyshev_field(self):
        
        def chebyshev_field(p, t):
            # p are the trainable parameters
            # t is the time

            # degrees = jnp.arange(self.n_basis)

            # legendre_polynomials, _ = jax.vmap(lambda d: lpmn_values(self.n_basis - 1, d, t, is_normalized = False))(degrees)
    
            chebyshev_polynomials = [Perceptron.chebyshev_polynomial(i, t) for i in range(self.n_basis)]

            chebyshev_polynomials = jnp.array(chebyshev_polynomials)

            return jnp.dot(p, chebyshev_polynomials)
        
        return chebyshev_field
    
    # def get_piecewise_constant_field(self):

    #     return qml.pulse.pwc(self.timespan)

    def get_piecewise_constant_field(self):

        def pwc_field(p,t):

            step_functions = jnp.array([Perceptron.step_function(i, t) for i in jnp.linspace(0,self.timespan,self.n_basis+2)[1:-1]])
        
            return jnp.dot(p, step_functions)
        
        return pwc_field
    
    def get_delta_field(self):

        # if not self.field_basis_functions:

        #     self.field_basis_functions = [jax.jit(lambda t : Perceptron.delta_function(i, t)) for i in jnp.linspace(0,self.timespan,self.n_basis+2)[1:-1]]

        def delta_field(p,t):

            delta_functions = jnp.array([Perceptron.delta_function(i, t) for i in jnp.linspace(0,self.timespan,self.n_basis+2)[1:-1]])
            # delta_functions = jnp.array([Perceptron.delta_function(i, t) for i in jnp.linspace(0,self.timespan,self.n_basis)])

            return jnp.dot(p, delta_functions)

        return delta_field
    
    def get_gaussian_field(self):

        def gaussian_field(p,t):

            delta_functions = jnp.array([Perceptron.gaussian(i, t, self.pulse_width) for i in jnp.linspace(0,self.timespan,self.n_basis+2)[1:-1]])
            # delta_functions = jnp.array([Perceptron.delta_function(i, t) for i in jnp.linspace(0,self.timespan,self.n_basis)])

            return jnp.dot(p, delta_functions)

        return gaussian_field



    def vector_to_hamiltonian_parameters(self, param_vector):

        n_js = self.n_qubits - 1

        n_vs = 2 * self.n_qubits * self.n_basis

        js_vector = param_vector[:n_js]
        js_vector = jnp.array([1 for i in range(n_js)])

        vs_matrix = param_vector[n_js:].reshape((-1,self.n_basis))

        return list(js_vector) + list(vs_matrix)

        # return list(param_vector)

    def get_random_parameter_vector(self, seed):

        n_params = self.n_qubits - 1 + 2 * self.n_qubits * self.n_basis

        # n_params = 2 * self.n_qubits * self.n_basis

        key = jax.random.PRNGKey(seed)

        param_vector = jax.random.uniform(key, shape = [n_params])

        return param_vector
    
    def get_1d_ising_hamiltonian(self, transverse_field_coefficient):

        obs = [qml.PauliZ(i) @ qml.PauliZ(i+1) for i in range((self.n_qubits-1))]
        coeffs = [1.0 for i in range((self.n_qubits-1))]
        z_term = qml.Hamiltonian(coeffs, obs)

        obs = [qml.PauliX(i) for i in range((self.n_qubits))]
        coeffs = [transverse_field_coefficient for i in range((self.n_qubits))]
        transverse_field_term = qml.Hamiltonian(coeffs, obs)

        ising_hamiltonian = transverse_field_term + z_term

        if self.n_qubits < 5:

            n_eigenvalues = 2**(self.n_qubits-1)

        else:
            n_eigenvalues = 30

        energies = qml.eigvals(ising_hamiltonian, k=n_eigenvalues, which='SR')

        return ising_hamiltonian, energies

    @staticmethod
    def legendre_polynomial(n, x):
        if n == 0:
            return jnp.ones_like(x)
        elif n == 1:
            return x
        else:
            p_prev = jnp.ones_like(x)
            p = x
            for _ in range(2, n + 1):
                p_prev, p = p, ((2 * _) - 1) / _ * x * p - (_ - 1) / _ * p_prev
            return p
        
    @staticmethod
    def chebyshev_polynomial(n, x):
        if n == 0:
            return jnp.ones_like(x)
        elif n == 1:
            return x
        else:
            return 2 * x * Perceptron.chebyshev_polynomial(n - 1, x) - Perceptron.chebyshev_polynomial(n - 2, x)

    @staticmethod
    def step_function(threshold, x):

        return jnp.where(x >= threshold, 1.0, 0.0)
    
    @staticmethod
    def delta_function(x0, x, epsilon=0.5e-2):

        return jnp.where(jnp.abs(x - x0) < epsilon, 1.0 / epsilon, 0.0)
        # return jnp.where(jnp.abs(x - x0) < epsilon, 1.0, 0.0)
    
    @staticmethod
    def gaussian(mean, x, std_dev=0.5e-2):
        return jnp.exp(-(x - mean)**2 / (2 * std_dev**2)) / (std_dev * jnp.sqrt(2 * jnp.pi))




class NativePerceptron():

    def __init__(self, n_qubits, n_basis, timespan = 1,  basis='fourier', pulse_width = 0.5e-2, native_coupling = 1.0):
    
        self.n_qubits = n_qubits
        self.n_basis = n_basis
        self.basis = basis
        self.pulse_width = pulse_width
        self.timespan = timespan
        self.native_coupling = native_coupling
        # self.field_basis_functions = []

        self.create_parametrized_hamiltonian()



    def create_parametrized_hamiltonian(self):

        self.create_native_hamiltonian()
        self.create_control_hamiltonian()
        
        self.H = self. H_native + self.H_control 
        

    def create_native_hamiltonian(self):

        # self.native_fields = [self.get_constant_field() for i in range(self.n_qubits-1)]
        self.native_fields = [self.native_coupling for i in range(self.n_qubits-1)]
        self.H_native_operators = [qml.PauliZ(i) @ qml.PauliZ(self.n_qubits-1) for i in range(self.n_qubits-1)]

        self.H_native = qml.dot(self.native_fields, self.H_native_operators)


    def create_control_hamiltonian(self):

        if self.basis == 'fourier':
            self.control_fields_x = [self.get_fourier_field() for i in range(self.n_qubits)]
            self.control_fields_y = [self.get_fourier_field() for i in range(self.n_qubits)]

        if self.basis == 'legendre':
            self.control_fields_x = [self.get_legendre_field() for i in range(self.n_qubits)]
            self.control_fields_y = [self.get_legendre_field() for i in range(self.n_qubits)]

        if self.basis == 'chebyshev':
            self.control_fields_x = [self.get_chebyshev_field() for i in range(self.n_qubits)]
            self.control_fields_y = [self.get_chebyshev_field() for i in range(self.n_qubits)]

        if self.basis == 'pwc':
            self.control_fields_x = [self.get_piecewise_constant_field() for i in range(self.n_qubits)]
            self.control_fields_y = [self.get_piecewise_constant_field() for i in range(self.n_qubits)]

        if self.basis == 'delta':
            
            self.field_basis_functions = [jax.jit(lambda t : Perceptron.delta_function(i, t)) for i in jnp.linspace(0,self.timespan,self.n_basis+2)[1:-1]]

            self.control_fields_x = [self.get_delta_field() for i in range(self.n_qubits)]
            self.control_fields_y = [self.get_delta_field() for i in range(self.n_qubits)]

        if self.basis == 'gaussian':

            self.field_basis_functions = [jax.jit(lambda t : Perceptron.gaussian(i, t, self.pulse_width)) for i in jnp.linspace(0,self.timespan,self.n_basis+2)[1:-1]]

            self.control_fields_x = [self.get_gaussian_field() for i in range(self.n_qubits)]
            self.control_fields_y = [self.get_gaussian_field() for i in range(self.n_qubits)]


        self.H_control_operators_x = [qml.PauliX(i) for i in range(self.n_qubits)]
        self.H_control_operators_y = [qml.PauliY(i) for i in range(self.n_qubits)]

        self.control_fields = self.control_fields_x + self.control_fields_y
        self.H_control_operators = self.H_control_operators_x + self.H_control_operators_y

        self.H_control = qml.dot(self.control_fields, self.H_control_operators)


    def get_constant_field(self):

        def constant_field(p,t):
            # p is the trainable parameter
            # t is the time
            return p

        return constant_field


    def get_fourier_field(self):
        
        def fourier_field(p, t):
            # p are the trainable parameters
            # t is the time
            return jnp.dot(p, jnp.cos(2*jnp.pi*(t)*jnp.arange(self.n_basis))) # QUESTION: WHY ARE WE MULTIPLYING BY 2?
        
        return fourier_field
    

    def get_legendre_field(self):
        
        def legendre_field(p, t):
            # p are the trainable parameters
            # t is the time

            # degrees = jnp.arange(self.n_basis)

            # legendre_polynomials, _ = jax.vmap(lambda d: lpmn_values(self.n_basis - 1, d, t, is_normalized = False))(degrees)
    
            legendre_polynomials = [Perceptron.legendre_polynomial(i, t) for i in range(self.n_basis)]

            legendre_polynomials = jnp.array(legendre_polynomials)

            return jnp.dot(p, legendre_polynomials)
        
        return legendre_field
    

    def get_chebyshev_field(self):
        
        def chebyshev_field(p, t):
            # p are the trainable parameters
            # t is the time

            # degrees = jnp.arange(self.n_basis)

            # legendre_polynomials, _ = jax.vmap(lambda d: lpmn_values(self.n_basis - 1, d, t, is_normalized = False))(degrees)
    
            chebyshev_polynomials = [Perceptron.chebyshev_polynomial(i, t) for i in range(self.n_basis)]

            chebyshev_polynomials = jnp.array(chebyshev_polynomials)

            return jnp.dot(p, chebyshev_polynomials)
        
        return chebyshev_field
    
    # def get_piecewise_constant_field(self):

    #     return qml.pulse.pwc(self.timespan)

    def get_piecewise_constant_field(self):

        def pwc_field(p,t):

            step_functions = jnp.array([Perceptron.step_function(i, t) for i in jnp.linspace(0,self.timespan,self.n_basis+2)[1:-1]])
        
            return jnp.dot(p, step_functions)
        
        return pwc_field
    
    def get_delta_field(self):

        # if not self.field_basis_functions:

        #     self.field_basis_functions = [jax.jit(lambda t : Perceptron.delta_function(i, t)) for i in jnp.linspace(0,self.timespan,self.n_basis+2)[1:-1]]

        def delta_field(p,t):

            delta_functions = jnp.array([Perceptron.delta_function(i, t) for i in jnp.linspace(0,self.timespan,self.n_basis+2)[1:-1]])
            # delta_functions = jnp.array([Perceptron.delta_function(i, t) for i in jnp.linspace(0,self.timespan,self.n_basis)])

            return jnp.dot(p, delta_functions)

        return delta_field
    
    def get_gaussian_field(self):

        def gaussian_field(p,t):

            delta_functions = jnp.array([Perceptron.gaussian(i, t, self.pulse_width) for i in jnp.linspace(0,self.timespan,self.n_basis+2)[1:-1]])
            # delta_functions = jnp.array([Perceptron.delta_function(i, t) for i in jnp.linspace(0,self.timespan,self.n_basis)])

            return jnp.dot(p, delta_functions)

        return gaussian_field



    def vector_to_hamiltonian_parameters(self, param_vector):

        # n_js = self.n_qubits - 1

        n_vs = 2 * self.n_qubits * self.n_basis

        # js_vector = param_vector[:n_js]
        # # js_vector = jnp.array([1 for i in range(n_js)])

        # vs_matrix = param_vector[n_js:].reshape((-1,self.n_basis))
        vs_matrix = param_vector.reshape((-1,self.n_basis))

        # return list(js_vector) + list(vs_matrix)
        return list(vs_matrix)

    def get_random_parameter_vector(self, seed):

        # n_params = self.n_qubits - 1 + 2 * self.n_qubits * self.n_basis

        n_params = 2 * self.n_qubits * self.n_basis

        key = jax.random.PRNGKey(seed)

        param_vector = jax.random.uniform(key, shape = [n_params])

        return param_vector
    
    def get_1d_ising_hamiltonian(self, transverse_field_coefficient):

        obs = [qml.PauliZ(i) @ qml.PauliZ(i+1) for i in range((self.n_qubits-1))]
        coeffs = [1.0 for i in range((self.n_qubits-1))]
        z_term = qml.Hamiltonian(coeffs, obs)

        obs = [qml.PauliX(i) for i in range((self.n_qubits))]
        coeffs = [transverse_field_coefficient for i in range((self.n_qubits))]
        transverse_field_term = qml.Hamiltonian(coeffs, obs)

        ising_hamiltonian = transverse_field_term + z_term

        if self.n_qubits < 5:

            n_eigenvalues = 2**(self.n_qubits-1)

        else:
            n_eigenvalues = 30

        energies = qml.eigvals(ising_hamiltonian, k=n_eigenvalues, which='SR')

        return ising_hamiltonian, energies

    @staticmethod
    def legendre_polynomial(n, x):
        if n == 0:
            return jnp.ones_like(x)
        elif n == 1:
            return x
        else:
            p_prev = jnp.ones_like(x)
            p = x
            for _ in range(2, n + 1):
                p_prev, p = p, ((2 * _) - 1) / _ * x * p - (_ - 1) / _ * p_prev
            return p
        
    @staticmethod
    def chebyshev_polynomial(n, x):
        if n == 0:
            return jnp.ones_like(x)
        elif n == 1:
            return x
        else:
            return 2 * x * Perceptron.chebyshev_polynomial(n - 1, x) - Perceptron.chebyshev_polynomial(n - 2, x)

    @staticmethod
    def step_function(threshold, x):

        return jnp.where(x >= threshold, 1.0, 0.0)
    
    @staticmethod
    def delta_function(x0, x, epsilon=0.5e-2):

        return jnp.where(jnp.abs(x - x0) < epsilon, 1.0 / epsilon, 0.0)
        # return jnp.where(jnp.abs(x - x0) < epsilon, 1.0, 0.0)
    
    @staticmethod
    def gaussian(mean, x, std_dev=0.5e-2):
        return jnp.exp(-(x - mean)**2 / (2 * std_dev**2)) / (std_dev * jnp.sqrt(2 * jnp.pi))
