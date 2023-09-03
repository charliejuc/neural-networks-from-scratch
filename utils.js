// Your vocabulary (usually built from your dataset)
const vocabulary = ["apple", "banana", "cherry", "date", "cat"];

// Your text description
const description = "cat cherry apple";

const tokensToNumberFactory = (vocabulary) => () => {
  // Tokenize and clean the text
  const tokens = description.toLowerCase().split(" ");

  // Populate the array based on the frequency of each word in the description
  const numericTokens = tokens.reduce((acc, token) => {
    const index = vocabulary.indexOf(token);
    if (index === -1) {
      return acc;
    }

    // Initialize an array of zeros based on the vocabulary size
    const tokens = Array(vocabulary.length).fill(0);
    ++tokens[index];

    acc.push(tokens);
    return acc;
  }, []);

  return numericTokens;
};

module.exports = {
  tokensToNumber: tokensToNumberFactory(vocabulary),
};
