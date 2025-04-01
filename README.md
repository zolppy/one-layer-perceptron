# One-Layer Perceptron (OLP)

![olp](https://github.com/user-attachments/assets/c149753c-50a1-4171-b1e3-c94958748f93)

Credit: image found on [image source](https://www.w3schools.com/ai/default.asp)

## 1. Perceptron Algorithm

The code implements the **Perceptron**, one of the earliest machine learning algorithms for binary classification. It is a linear classifier that:

-   Learns a decision boundary by adjusting weights iteratively;
-   Uses a threshold-based activation function (step function);
-   Updates weights only when misclassifications occur.

## 2. Model Parameters (Weights and Bias)

-   **Weights**: Learned coefficients for input features (`self.weights`). They determine each feature's importance in decision-making;
-   **Bias**: Offset term (`self.bias`) allowing the model to shift decision boundary away from the origin.

## 3. Hyperparameters

-   **Learning Rate (`learning_rate`)**: Controls update magnitude during training. A smaller value (default = 0.0001) makes convergence slower but more stable;
-   **Epochs**: Number of full passes over the training data. Prevents infinite loops for non-separable data.

## 4. Activation Function (Step Function)

-   Converts weighted sum to binary output (0/1):

```python
def activation_function(self, weighted_sum: float) -> int:
    return 1 if weighted_sum >= 0 else 0
```

-   Creates a non-linear decision threshold while maintaining a linear decision boundary.

## 5. Training Loop and Epochs

-   **Epoch**: Full iteration over the entire dataset (outer loop in `train`);
-   **Early Stopping**: Checks `converged` flag to terminate early if no errors occur.

## 6. Forward Pass and Prediction

-   Compute weighted sum:

```python
weighted_sum = sum(x * w for x, w in zip(inputs, self.weights)) + self.bias
```

-   Apply activation function to get prediction (0/1).

## 7. Error-Driven Learning

-   **Error Calculation**:

```python
error = target - prediction
```

-   **Weight Update Rule** (Perceptron Learning Rule):

```python
self.weights[i] += self.learning_rate * error * inputs[i]
self.bias += self.learning_rate * error
```

## 8. Online Learning

-   Updates weights **immediately after each sample** rather than using batch updates.

## 9. Convergence

-   **Perceptron Convergence Theorem**: Guarantees convergence to a solution **if data is linearly separable**;
-   **Implementation**: `converged` flag checks if all samples are correctly classified in an epoch.

## 10. Linear Separability

-   The perceptron can **only learn linearly separable patterns**;
-   **Decision Boundary**: Defined by `w·x + b = 0`, a hyperplane in the input space.

## 11. Bias Term

-   Acts as a trainable offset, equivalent to a weight for a constant input of 1.

## 12. Zero Initialization

-   Weights and bias start at zero (`[0.0] * num_inputs, bias: float = 0.0`).

## 13. Supervised Learning Framework

-   Requires labeled data (`dataset` and `desired_outputs`);
-   Learns by comparing predictions to ground truth.

## 14. Inference Phase

-   **Prediction**: Uses learned weights/bias on new data via `predict()` method.

## 15. Limitations

1. **Binary Classification Only**: Outputs 0/1 via step function;
2. **Linear Boundaries**: Cannot handle complex/non-linear patterns;
3. **Sensitivity to Learning Rate**: Poor choice can cause slow convergence or oscillations.

## References

- [Machine Learning](https://www.w3schools.com/ai/default.asp)
