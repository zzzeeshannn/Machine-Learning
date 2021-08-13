import numpy as np
from matplotlib import pyplot as plt

# Define the function here
def objective(x):
    return x**2

# Define the derivative of the function here
def derivative(x):
    return 2*x

# Define gradient descent here
def gradient_descent(objective, derivative, bounds, iter, step_size):
    """
    :param objective: The objective function
    :param derivative: First order derivative of the function
    :param bounds: Range for the input
    :param iter: Number of iterations of gradient descent
    :param step_size: Step_size for gradient descent

    :return: x and y values of the gd movement
    """
    # Declare empty lists to store the inputs and respective outputs
    inputs = list()
    results = list()

    # Generate an initial point for the gradient descent to start
    # input = -1
    input = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    # Run the gradient descent over number of iteration
    for _ in range(iter):
        # Calculate gradient i.e first order derivative of the objective function
        gradient = derivative(input)
        # Take a step
        input -= step_size*gradient
        # Get the corresponding value with the objective function
        output = objective(input)
        # Store both
        inputs.append(input)
        results.append(output)

    return [inputs, results]

# Define main here
def main():
    # Define the bounds of the function
    bounds = np.asarray([[-1.0, 1.0]])
    # Define the total number of iterations and the step size for gradient descent
    iter = 50
    step_size = 0.1
    # Call gradient descent here
    inputs, results = gradient_descent(objective, derivative, bounds, iter, step_size)
    # Sample input with +0.1 increments
    input = np.arange(bounds[0,0], bounds[0,1] + 0.1, 0.1)
    # Compute targets
    result = objective(input)
    # Plot together
    plt.plot(input, result)
    plt.plot(inputs, results, '.-', color='red')
    plt.show()

    return 0

if __name__ == '__main__':
    main()