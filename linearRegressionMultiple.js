const math = require("mathjs");

class MultipleLinearRegression {
  constructor() {
    this.coefficients = [];
  }

  fit(X, y) {
    const XT = math.transpose(X);
    const XT_X = math.multiply(XT, X);
    const inverse = math.inv(XT_X);
    const XT_y = math.multiply(XT, y);
    const beta = math.multiply(inverse, XT_y);
    this.coefficients = beta;
  }

  predict(X) {
    const yPredicted = [];
    for (let i = 0; i < X.length; i++) {
      let y = this.coefficients[0]; // Assuming the first coefficient is the intercept
      for (let j = 1; j < this.coefficients.length; j++) {
        y += this.coefficients[j] * X[i][j - 1];
      }
      yPredicted.push(y);
    }
    return yPredicted;
  }
}

const yIntercept = 30; // we set this value because we have few data, this value should be 1.

const X = [
  [yIntercept, 170, 30],
  [yIntercept, 165, 25],
  [yIntercept, 180, 40],
  [yIntercept, 150, 18],
  [yIntercept, 175, 34],
  [yIntercept, 160, 17],
  [yIntercept, 185, 45],
  [yIntercept, 145, 16],
  [yIntercept, 155, 20],
  [yIntercept, 190, 50],
  [yIntercept, 200, 55],
  [yIntercept, 120, 12],
  [yIntercept, 130, 15],
  [yIntercept, 140, 17],
  [yIntercept, 180, 20],
  [yIntercept, 160, 15],
  [yIntercept, 160, 45],
  [yIntercept, 210, 37],
  [yIntercept, 170, 53],
  [yIntercept, 176, 27],
  [yIntercept, 170, 30],
  [yIntercept, 165, 25],
  [yIntercept, 180, 40],
  [yIntercept, 150, 18],
  [yIntercept, 175, 34],
  [yIntercept, 160, 22],
  [yIntercept, 185, 45],
  [yIntercept, 145, 16],
  [yIntercept, 155, 20],
  [yIntercept, 190, 50],
  [yIntercept, 200, 55],
  [yIntercept, 120, 12],
  [yIntercept, 130, 15],
  [yIntercept, 140, 17],
  [yIntercept, 180, 20],
  [yIntercept, 157, 20],
];

const y = [
  70, // 70 kg
  65, // 65 kg
  75, // 75 kg
  50, // 50 kg
  72, // 72 kg
  55, // 55 kg
  85, // 85 kg
  45, // 45 kg
  50, // 50 kg
  100, // 100 kg
  95, // 95 kg
  36, // 36 kg
  40, // 40 kg
  43, // 43 kg
  81, // 81 kg
  62, // 62 kg
  70, // 70 kg
  96, // 96 kg
  76, // 76 kg
  76, // 76 kg
  68, // 68 kg
  65, // 65 kg
  80, // 80 kg
  45, // 45 kg
  75, // 75 kg
  55, // 55 kg
  85, // 85 kg
  45, // 45 kg
  52, // 52 kg
  90, // 90 kg
  95, // 95 kg
  36, // 36 kg
  40, // 40 kg
  43, // 43 kg
  81, // 81 kg
  59, // 59 kg
];

// Create and train the model
const mlr = new MultipleLinearRegression();
mlr.fit(X, y);

// Make predictions
const new_X = [
  [yIntercept, 185, 40],
  [yIntercept, 175, 35],
  [yIntercept, 160, 20],
];
const predictions = mlr.predict(new_X);

console.log(predictions); // Output: [estimated weight for 175 cm/35 year old, estimated weight for 160 cm/20 year old]
