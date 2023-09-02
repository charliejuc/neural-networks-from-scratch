const { NeuralNetwork } = require("./NeuralNetwork2");

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
