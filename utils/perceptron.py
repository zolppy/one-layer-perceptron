from typing import List, Union

class Perceptron:
    """A simple Perceptron classifier implementing the step activation function."""
    
    def __init__(self, num_inputs: int, learning_rate: float = 0.0001) -> None:
        """Initialize the Perceptron with given parameters.

        Args:
            num_inputs (int): Number of input features.
            learning_rate (float, optional): Learning rate for weight updates. Defaults to 0.0001.
        """
        self.weights: List[float] = [0.0] * num_inputs  # Initialize weights as zeros
        self.bias: float = 0.0  # Initialize bias to zero
        self.learning_rate: float = learning_rate  # Learning rate for training

    def activation_function(self, weighted_sum: float) -> int:
        """Apply the Heaviside step function as the activation function.

        Args:
            weighted_sum (float): Weighted sum of inputs and weights plus bias.

        Returns:
            int: 1 if weighted_sum >= 0, 0 otherwise.
        """
        return 1 if weighted_sum >= 0 else 0

    def train(self, dataset: List[List[Union[float, int]]], desired_outputs: List[int], epochs: int = 100) -> None:
        """Train the Perceptron on the dataset.

        Iterates over the dataset for up to `epochs` times, adjusting weights and bias 
        to minimize prediction errors. Stops early if all predictions are correct.

        Args:
            dataset (List[List[Union[float, int]]]): Training data where each sample is a list of features.
            desired_outputs (List[int]): Target labels (0 or 1) corresponding to each sample.
            epochs (int, optional): Maximum number of training iterations. Defaults to 100.
        """
        for _ in range(epochs):
            converged = True  # Flag to check if all predictions are correct
            
            # Iterate over each sample in the dataset
            for inputs, target in zip(dataset, desired_outputs):
                # Calculate weighted sum of inputs and weights, add bias
                weighted_sum = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias
                prediction = self.activation_function(weighted_sum)
                error = target - prediction  # Compute error (0, 1, or -1)
                
                if error != 0:
                    converged = False  # Mark as not converged
                    # Update weights and bias using the perceptron update rule
                    for i in range(len(self.weights)):
                        self.weights[i] += self.learning_rate * error * inputs[i]
                    self.bias += self.learning_rate * error
                    
            # Early stopping if no errors in the entire epoch
            if converged:
                break

    def predict(self, inputs: List[Union[float, int]]) -> int:
        """Predict the output class for a given input sample.

        Args:
            inputs (List[Union[float, int]]): Input features to predict.

        Returns:
            int: Predicted class (0 or 1).
        """
        # Calculate weighted sum of inputs and weights, add bias
        weighted_sum = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias
        return self.activation_function(weighted_sum)