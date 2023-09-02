const fs = require("fs");
const math = require("mathjs");

class Layer {
  constructor(inputNodes, outputNodes) {
    this.inputNodes = inputNodes;
    this.outputNodes = outputNodes;
    this.weights = math.random([this.outputNodes, this.inputNodes], -1, 1);
    this.bias = math.zeros([this.outputNodes, 1]);
  }

  updateWeights(weightDeltas) {
    this.weights = math.add(this.weights, weightDeltas);
  }

  updateBias(biasDeltas) {
    this.bias = math.add(this.bias, biasDeltas);
  }

  calculateErrors(nextLayerWeights, nextLayerErrors) {
    return math.multiply(math.transpose(nextLayerWeights), nextLayerErrors);
  }

  forward(inputs, activationFunction) {
    let output = math.add(math.multiply(this.weights, inputs), this.bias);
    return math.map(output, activationFunction);
  }

  saveToData() {
    return {
      inputNodes: this.inputNodes,
      outputNodes: this.outputNodes,
      weights: this.weights.toArray(),
      bias: this.bias.toArray(),
    };
  }

  loadFromData(layerData) {
    this.inputNodes = layerData.inputNodes;
    this.outputNodes = layerData.outputNodes;
    this.weights = math.matrix(layerData.weights);
    this.bias = math.matrix(layerData.bias);
  }
}

class NeuralNetwork {
  constructor(inputNodes, hiddenNodes, outputNodes) {
    this.inputLayer = new Layer(inputNodes, hiddenNodes);
    this.outputLayer = new Layer(hiddenNodes, outputNodes);
    this.learningRate = 0.09;
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
    let hidden = this.inputLayer.forward(inputs, this.sigmoid);
    let outputs = this.outputLayer.forward(hidden, this.sigmoid);
    return { inputs, hidden, outputs };
  }

  calculateErrors(outputs, targetArray) {
    let targets = math.matrix(targetArray);
    let outputErrors = math.subtract(targets, outputs);
    let hiddenErrors = this.inputLayer.calculateErrors(
      this.outputLayer.weights,
      outputErrors
    );

    return { outputErrors, hiddenErrors };
  }

  gradientDescentOutput(hidden, gradients) {
    let hiddenT = math.transpose(hidden);
    let weight_ho_deltas = math.multiply(gradients, hiddenT);
    this.outputLayer.updateWeights(weight_ho_deltas);
    this.outputLayer.updateBias(gradients);
  }

  gradientDescentHidden(inputs, hiddenGradient) {
    let inputsT = math.transpose(inputs);
    let weight_ih_deltas = math.multiply(hiddenGradient, inputsT);
    this.inputLayer.updateWeights(weight_ih_deltas);
    this.inputLayer.updateBias(hiddenGradient);
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

  saveToFile(filename) {
    const data = {
      learningRate: this.learningRate,
      inputLayer: this.inputLayer.saveToData(),
      outputLayer: this.outputLayer.saveToData(),
    };
    fs.writeFileSync(filename, JSON.stringify(data, null, 2));
  }

  loadFromFile(filename) {
    const data = JSON.parse(fs.readFileSync(filename, "utf8"));
    this.learningRate = data.learningRate;
    this.inputLayer.loadFromData(data.inputLayer);
    this.outputLayer.loadFromData(data.outputLayer);
  }
}

module.exports = { NeuralNetwork };
