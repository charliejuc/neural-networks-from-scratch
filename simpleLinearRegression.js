class SimpleLinearRegression {
  constructor() {
    this.beta0 = 0; // y-intercept
    this.beta1 = 0; // slope of the line
  }

  fit(X, y) {
    const n = X.length;
    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumX2 = 0;

    for (let i = 0; i < n; i++) {
      sumX += X[i];
      sumY += y[i];
      sumXY += X[i] * y[i];
      sumX2 += X[i] * X[i];
    }

    const numerator = n * sumXY - sumX * sumY;
    const denominator = n * sumX2 - sumX ** 2;

    this.beta1 = numerator / denominator;
    this.beta0 = (sumY - this.beta1 * sumX) / n;
  }

  predict(x) {
    return this.beta0 + this.beta1 * x;
  }
}

// const sizes = [1500, 1800, 2100, 2400, 2700]; // X values
// const prices = [250, 275, 300, 320, 350]; // y values

// const regression = new SimpleLinearRegression();
// regression.fit(sizes, prices);

// const predicted = regression.predict(1700);
// console.log(`Predicted house price: $${predicted * 1000}`);

// Sample data: pushStrength vs slideDistance
const pushStrengths = [2, 4, 5, 7, 9]; // How hard kids pushed their toy cars.
const slideDistances = [5, 10, 12, 18, 22]; // How far the cars slid.

const regression = new SimpleLinearRegression();
regression.fit(pushStrengths, slideDistances);

// Predicting the slide distance for a given push strength
const pushStrengthForPrediction = 6;
const predictedSlideDistance = regression.predict(pushStrengthForPrediction);
console.log(
  `For a push strength of ${pushStrengthForPrediction}, the predicted slide distance is ${predictedSlideDistance.toFixed(
    2
  )} units.`
);

// TODO: Linear regression with several variables
