import pickle
import jax.numpy as jnp

def load_simulation_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def print_simulation_results(data):
    final_energy = data['energies'][-1]
    final_hessian = data['final_hessian']
    hessian_eigenvalues = data['final_hessian_eigenvalues']

    print("Final Energy Found:", final_energy)
    print("Final Hessian Matrix:\n", final_hessian)
    print("Eigenvalues of the Final Hessian:\n", hessian_eigenvalues)

def main():
    file_path = 'simulation_data.pkl'  # Update this with the correct file path
    simulation_data = load_simulation_data(file_path)
    print_simulation_results(simulation_data)

if __name__ == "__main__":
    main()
