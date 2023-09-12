const { NeuralNetwork } = require("./NeuralNetwork");

const nn = new NeuralNetwork(4, 4, 1);

const threshold = 0.3;
const categories = [
  ["Fruit", [1, 0, 0, 0]],
  ["Animal", [0, 1, 0, 0]],
  ["Vehicle", [0, 0, 1, 0]],
  ["Computer", [0, 0, 0, 1]],
];

// Sample training data
const training_data = [
  { input: [0.2, 0.2, 0.8, 0], target: categories[0][1] },
  { input: [0.1, 0.2, 0.7, 0], target: categories[0][1] },
  { input: [0.1, 0.1, 0.9, 0], target: categories[0][1] },
  { input: [0.2, 0.15, 0.81, 0], target: categories[0][1] },

  { input: [0.7, 0.2, 0.3, 0], target: categories[1][1] },
  { input: [0.65, 0.25, 0.27, 0], target: categories[1][1] },
  { input: [0.9, 0.18, 0.2, 0], target: categories[1][1] },
  { input: [0.6, 0.1, 0.1, 0], target: categories[1][1] },

  { input: [0.1, 0.8, 0.2, 0], target: categories[2][1] },
  { input: [0.2, 0.9, 0.1, 0], target: categories[2][1] },
  { input: [0.1, 0.75, 0.2, 0], target: categories[2][1] },
  { input: [0.1, 0.6, 0.3, 0], target: categories[2][1] },

  { input: [0.6, 0.1, 0.6, 0], target: categories[3][1] },
  { input: [0.8, 0.2, 0.7, 0], target: categories[3][1] },
  { input: [0.7, 0.2, 0.8, 0], target: categories[3][1] },
  { input: [0.7, 0.15, 0.7, 0], target: categories[3][1] },
];

// nn.loadFromFile("neural_network_state.json");

// Train the neural network
// const epochs = 20000;
// for (let i = 0; i < epochs; i++) {
//   for (let data of training_data) {
//     nn.train(data.input, data.target);
//   }
// }

// Test the neural network with a new description
const newDescriptions = [
  [0.2, 0.2, 0.8, 0], // This should be close to "Fruit"
  [0.8, 0.2, 0.3, 0], // This should be close to "Animal"
  [0.1, 0.8, 0.3, 0], // This should be close to "Vehicle"
  [0.7, 0.2, 0.6, 0], // This should be close to "Computer"

  [0.12, 0.13, 0.6, 0], // This should be close to "Fruit"
  [0.73, 0.15, 0.27, 0], // This should be close to "Animal"
  [0.4, 0.6, 0.38, 0], // This should be close to "Vehicle"
  [0.73, 0.1, 0.69, 0], // This should be close to "Computer"
];

newDescriptions.forEach((newDescription, i) => {
  const prediction = nn.predict(newDescription);
  // i === 2 && console.log(prediction);
  const maxPrediction = Math.max(...prediction[0]);
  const parsedPrediction = prediction[0].map((p) => {
    if (p >= threshold && p === maxPrediction) {
      return 1;
    }

    return 0;
  });
  console.log("Parsed prediction is:", parsedPrediction);

  // Interpret the prediction to get the category
  const predictedCategory = categories.find((category) => {
    return category[1].every((v, i) => v === parsedPrediction[i]);
  });
  console.log(
    `Predicted category: ${
      predictedCategory ? predictedCategory[0] : "No match"
    }`
  );
});

// nn.saveToFile("neural_network_state2.json");
