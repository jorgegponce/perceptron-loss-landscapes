# Notes on this project so far

Idea: we are interested in studying the loss landscape of these new quantum perceptron to better understand its capabilities and important things such as how trainable and reliable this new approach to implementing a perceptron is. 

For this, we are interested in quantifying how ...


## Log:

### Analog Simulations

- Currently waiting for the Cluster people to reproduce my error and figure out what is wrong with the simulation scripts

### Digitlal Simulations:

#### Tensorly:
___
Didn't work. I tried to impelemt CZ gates between non-adjacent qubits but kept getting non-sensical results (0 state vectors or state vectors that were not normalized). I reached to Taylor to inquire about this and she told me this is not possible to do with Tensorly-quantum as it is. I needed to implement a new type of CZ gate.

I decided to not bother anyone and does, after some research on fast and easy to use quantum simulators, I landed on Qulacs

#### Qulacs:
___
I was able to succesfully implement the VQEs with the right perceptron circuit (I forgot to apply CZ gates between the qubits, so I actaully had no entangling layers. But this was not a problem since we had to migrate once more to a different simulator).

The simulations are fast and reliable, but when trying to calculate the Hessian matrices with rexpect to the parameters, I could not this functionality implemented in Qulacs and instead was estimating the hessian using finite differences with the ORQVIZ get_Hessian function. This takes a really long time and are usually not numerically stable. In the end, because the finite difference method this function implements indeed turned out to be very unstable for the analog case, we decided to migrate to a different simulator/framework that could calculate the exact Hessians through Automatic Differentiation: Pennylane

#### Pennylane:
___







## Submitting jobs to the cluster
________


To automize the creation of job arrays, I created a perceptron_template.sbacth file which acts as a template for the submission of all the different types of perceptrons. It populates some environment variables inside it with set values via the `envsubst` unix program and then saves the file. Here is an example of how to use it:

To create the SBATCH file to run a 4-qubit perceptron simulation with a 0.1 transverse field coefficient, for a given number of days/hours/minutes, I run the following command:

```
export QUBITS=4 FIELD=0.1 DAYS=0 HOURS=00 MINUTES=10 && envsubst '$QUBITS, $FIELD, $DAYS, $HOURS, $MINUTES' < perceptron_template.sbatch > perceptron_${QUBITS}_qubits_${FIELD}_field.sbatch

```

The command will create a file with the name `perceptron_${QUBITS}_qubits_${FIELD}_field.sbatch` on the same directory as the template, and it will have populated the specified environment variables with the assigned values.
