const math = require("mathjs");

class NeuralNetwork {
  constructor(inputNodes, hiddenNodes, outputNodes) {
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;

    // Initialize weights and biases
    this.weights_ih = math.random([this.hiddenNodes, this.inputNodes], -1, 1);
    this.weights_ho = math.random([this.outputNodes, this.hiddenNodes], -1, 1);
    this.bias_h = math.random([this.hiddenNodes, 1], -1, 1);
    this.bias_o = math.random([this.outputNodes, 1], -1, 1);

    this.learningRate = 0.1;
  }

  // Sigmoid activation function and its derivative
  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  dsigmoid(y) {
    return y * (1 - y);
  }

  forwardPropagation(inputArray) {
    let inputs = math.matrix(inputArray);
    let hidden = math.add(math.multiply(this.weights_ih, inputs), this.bias_h);
    hidden = math.map(hidden, this.sigmoid);

    let outputs = math.add(math.multiply(this.weights_ho, hidden), this.bias_o);
    outputs = math.map(outputs, this.sigmoid);

    return { inputs, hidden, outputs };
  }

  calculateErrors(outputs, targetArray) {
    let targets = math.matrix(targetArray);
    let outputErrors = math.subtract(targets, outputs);
    let hiddenErrors = math.multiply(
      math.transpose(this.weights_ho),
      outputErrors
    );

    return { outputErrors, hiddenErrors };
  }

  gradientDescentOutput(hidden, gradients) {
    let hiddenT = math.transpose(hidden);
    let weight_ho_deltas = math.multiply(gradients, hiddenT);
    this.weights_ho = math.add(this.weights_ho, weight_ho_deltas);
    this.bias_o = math.add(this.bias_o, gradients);
  }

  gradientDescentHidden(inputs, hiddenGradient) {
    let inputsT = math.transpose(inputs);
    let weight_ih_deltas = math.multiply(hiddenGradient, inputsT);
    this.weights_ih = math.add(this.weights_ih, weight_ih_deltas);
    this.bias_h = math.add(this.bias_h, hiddenGradient);
  }

  backwardPropagationOutput(hidden, outputs, outputErrors) {
    let gradients = math.map(outputs, this.dsigmoid);
    gradients = math.dotMultiply(gradients, outputErrors);
    gradients = math.multiply(gradients, this.learningRate);

    this.gradientDescentOutput(hidden, gradients);
  }

  backwardPropagationHidden(inputs, hidden, hiddenErrors) {
    let hiddenGradient = math.map(hidden, this.dsigmoid);
    hiddenGradient = math.dotMultiply(hiddenGradient, hiddenErrors);
    hiddenGradient = math.multiply(hiddenGradient, this.learningRate);

    this.gradientDescentHidden(inputs, hiddenGradient);
  }

  train(inputArray, targetArray) {
    const { inputs, hidden, outputs } = this.forwardPropagation(inputArray);
    const { outputErrors, hiddenErrors } = this.calculateErrors(
      outputs,
      targetArray
    );

    this.backwardPropagationHidden(inputs, hidden, hiddenErrors);
    this.backwardPropagationOutput(hidden, outputs, outputErrors);
  }

  predict(inputArray) {
    const { outputs } = this.forwardPropagation(inputArray);
    return outputs.toArray();
  }
}

const nn = new NeuralNetwork(2, 4, 1);

// Training data for XOR
const training_data = [
  { input: [[0], [0]], target: [0] },
  { input: [[0], [1]], target: [1] },
  { input: [[1], [0]], target: [1] },
  { input: [[1], [1]], target: [0] },
];

// Train the neural network
const epochs = 4000;
for (let i = 0; i < epochs; i++) {
  for (let data of training_data) {
    nn.train(data.input, data.target);
  }
}

// Test the neural network
for (let data of training_data) {
  const prediction = nn.predict(data.input);
  console.log(
    `Input: ${data.input} | Target: ${data.target} | Prediction: ${prediction}`
  );
}
