const math = require("mathjs");
const { NeuralNetwork, Layer } = require("./NeuralNetwork");

const nn = new NeuralNetwork([new Layer(3), new Layer(8), new Layer(6, 4)]);

const threshold = 0.7;
const categories = [
  ["Fruit", [1, 0, 0, 0]],
  ["Animal", [0, 1, 0, 0]],
  ["Vehicle", [0, 0, 1, 0]],
  ["Computer", [0, 0, 0, 1]],
];

// Sample training data
const training_data = [
  { input: [0.2, 0.2, 0.8], target: categories[0][1] },
  { input: [0.1, 0.2, 0.7], target: categories[0][1] },
  { input: [0.1, 0.1, 0.9], target: categories[0][1] },
  { input: [0.2, 0.15, 0.81], target: categories[0][1] },
  { input: [0.15, 0.22, 0.77], target: categories[0][1] },
  { input: [0.22, 0.3, 0.92], target: categories[0][1] },
  { input: [0.04, 0.08, 0.97], target: categories[0][1] },
  { input: [0.04, 0.08, 0.55], target: categories[0][1] },
  { input: [0.2, 0.2, 0.64], target: categories[0][1] },

  { input: [0.7, 0.2, 0.3], target: categories[1][1] },
  { input: [0.65, 0.25, 0.27], target: categories[1][1] },
  { input: [0.9, 0.18, 0.2], target: categories[1][1] },
  { input: [0.6, 0.1, 0.1], target: categories[1][1] },
  { input: [0.54, 0.08, 0.2], target: categories[1][1] },
  { input: [0.68, 0.22, 0.32], target: categories[1][1] },
  { input: [0.97, 0.33, 0.3], target: categories[1][1] },
  { input: [0.64, 0.1, 0.1], target: categories[1][1] },

  { input: [0.1, 0.8, 0.2], target: categories[2][1] },
  { input: [0.2, 0.9, 0.1], target: categories[2][1] },
  { input: [0.1, 0.75, 0.2], target: categories[2][1] },
  { input: [0.1, 0.6, 0.3], target: categories[2][1] },
  { input: [0.2, 0.99, 0.2], target: categories[2][1] },
  { input: [0.02, 0.9, 0.02], target: categories[2][1] },
  { input: [0.2, 0.55, 0.23], target: categories[2][1] },
  { input: [0.2, 0.67, 0.12], target: categories[2][1] },

  { input: [0.6, 0.1, 0.6], target: categories[3][1] },
  { input: [0.8, 0.2, 0.7], target: categories[3][1] },
  { input: [0.7, 0.2, 0.8], target: categories[3][1] },
  { input: [0.7, 0.15, 0.7], target: categories[3][1] },
  { input: [0.67, 0.15, 0.8], target: categories[3][1] },
  { input: [0.8, 0.1, 0.8], target: categories[3][1] },
  { input: [0.7, 0.2, 0.71], target: categories[3][1] },
  { input: [0.83, 0.34, 0.57], target: categories[3][1] },
];

nn.loadFromFile("basicCategorizationState.json");

// Train the neural network
// const epochs = 1000;
// for (let i = 0; i < epochs; i++) {
//   for (let data of training_data) {
//     nn.train(data.input, data.target);
//   }
// }

// Test the neural network with a new description
const newDescriptions = [
  [0.2, 0.2, 0.8], // This should be close to "Fruit"
  [0.8, 0.2, 0.3], // This should be close to "Animal"
  [0.1, 0.8, 0.3], // This should be close to "Vehicle"
  [0.7, 0.2, 0.6], // This should be close to "Computer"

  [0.12, 0.13, 0.6], // This should be close to "Fruit"
  [0.73, 0.15, 0.27], // This should be close to "Animal"
  [0.4, 0.6, 0.38], // This should be close to "Vehicle"
  [0.73, 0.1, 0.69], // This should be close to "Computer"
];

newDescriptions.forEach((newDescription) => {
  const prediction = nn.predict(newDescription);
  const parsedPrediction = prediction.map((p) => {
    const predicted = math.max(p);
    if (predicted >= threshold) {
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

nn.saveToFile("basicCategorizationState.json");
