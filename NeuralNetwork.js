const math = require("mathjs");
const fs = require("fs");

class Layer {
  constructor(inputNodes, outputNodes = null) {
    this.inputNodes = inputNodes;
    this.outputNodes = outputNodes;

    if (this.inputNodes && this.outputNodes) {
      this.initializeWeightsAndBiases();
    }
  }

  initializeWeightsAndBiases() {
    this.weights = math.matrix(
      math.random([this.outputNodes, this.inputNodes], -1, 1)
    );
    this.bias = math.matrix(math.zeros([this.outputNodes, 1]));
  }

  forward(inputs, activationFunction) {
    let output = math.add(math.multiply(this.weights, inputs), this.bias);
    return math.map(output, activationFunction);
  }

  backward(
    errors,
    outputs,
    prevLayerOutputs,
    activationDerivative,
    learningRate
  ) {
    let gradients = this.calculateGradient(
      errors,
      outputs,
      activationDerivative,
      learningRate
    );
    this.updateWeights(gradients, prevLayerOutputs);
  }

  calculateGradient(errors, outputs, activationDerivative, learningRate) {
    let gradients = math.map(outputs, activationDerivative);
    gradients = math.dotMultiply(gradients, errors);
    return math.multiply(gradients, learningRate);
  }

  calculateOutputErrors(targets, currentOutput) {
    let reshapedTargets = math.reshape(targets, [this.outputNodes, 1]);

    return math.subtract(reshapedTargets, currentOutput);
  }

  updateWeights(gradients, prevLayerOutputs) {
    let deltas = math.multiply(gradients, math.transpose(prevLayerOutputs));
    this.weights = math.add(this.weights, deltas);
    this.bias = math.add(this.bias, gradients);
  }

  saveToData() {
    return {
      inputNodes: this.inputNodes,
      outputNodes: this.outputNodes,
      weights: this.weights.toArray(),
      bias: this.bias.toArray(),
    };
  }

  loadFromData(data) {
    this.inputNodes = data.inputNodes;
    this.outputNodes = data.outputNodes;
    this.weights = math.matrix(data.weights);
    this.bias = math.matrix(data.bias);
  }
}

class NeuralNetwork {
  constructor(layers) {
    this.layers = layers;
    this.learningRate = 0.05;

    // Automatically set the output nodes for each layer based on the next layer's input nodes
    for (let i = 0; i < this.layers.length - 1; i++) {
      this.layers[i].outputNodes = this.layers[i + 1].inputNodes;
      this.layers[i].initializeWeightsAndBiases();
    }
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  dsigmoid(y) {
    return y * (1 - y);
  }

  train(inputArray, targetArray) {
    let inputs = math.matrix(inputArray);
    let targets = math.matrix(targetArray);

    // Forward propagation through all layers
    let currentOutput = inputs;
    const outputs = [];
    for (const layer of this.layers) {
      currentOutput = layer.forward(currentOutput, this.sigmoid);
      outputs.push(currentOutput);
    }

    // Calculate errors for the output layer
    let outputErrors = this.layers[
      this.layers.length - 1
    ].calculateOutputErrors(targets, currentOutput);

    // Backward propagation through all layers
    let nextLayerErrors = outputErrors;
    for (let i = this.layers.length - 1; i > 0; i--) {
      const layer = this.layers[i];
      const output = outputs[i];
      const prevLayerOutput = outputs[i - 1];

      layer.backward(
        nextLayerErrors,
        output,
        prevLayerOutput,
        this.dsigmoid,
        this.learningRate
      );

      nextLayerErrors = math.multiply(
        math.transpose(layer.weights),
        nextLayerErrors
      );
    }
  }

  predict(inputArray) {
    let currentOutput = math.matrix(inputArray);
    for (const layer of this.layers) {
      currentOutput = layer.forward(currentOutput, this.sigmoid);
    }

    return currentOutput.toArray();
  }

  saveToFile(filename) {
    const data = {
      learningRate: this.learningRate,
      layers: this.layers.map((layer) => layer.saveToData()),
    };
    fs.writeFileSync(filename, JSON.stringify(data, null, 2));
  }

  loadFromFile(filename) {
    const data = JSON.parse(fs.readFileSync(filename, "utf8"));
    this.learningRate = data.learningRate;
    for (let i = 0; i < this.layers.length; i++) {
      this.layers[i].loadFromData(data.layers[i]);
    }
  }
}

module.exports = { Layer, NeuralNetwork };
