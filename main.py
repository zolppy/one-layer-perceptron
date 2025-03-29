from utils.perceptron import Perceptron

if __name__ == "__main__":
    dataset = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    desired_outputs = [0, 0, 0, 1]

    perceptron = Perceptron(
        num_inputs=2
    )
    perceptron.train(
        dataset=dataset,
        desired_outputs=desired_outputs,
        epochs=100
    )

    print(f"Desired_output: {desired_outputs[3]}, output: {perceptron.predict(dataset[3])}")