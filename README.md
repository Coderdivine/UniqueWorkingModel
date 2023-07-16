# UniqueWorkingModel

This repository contains a Jupyter Notebook file (`WorkingModel.ipynb`) that showcases a unique working model for word prediction using a Long Short-Term Memory (LSTM) neural network. The model is trained on a dataset to predict unique words based on the sentences provided.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Predicting Unique Words](#predicting-unique-words)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used for training the word prediction model is expected to be in JSON format. Each data point in the dataset consists of a sentence and a list of unique words present in that sentence. The dataset JSON file should be placed in the same directory as the `WorkingModel.ipynb` notebook.

## Installation

To use this working model, you need to have Python installed along with the required dependencies. You can install the dependencies by running the following command in your terminal or command prompt:

```shell
!pip install -r requirements.txt
```

## Usage

To use the word prediction model, follow these steps:

1. Clone or download this repository to your local machine.
2. Install the required dependencies using the command mentioned in the [Installation](#installation) section.
3. Place your dataset JSON file in the same directory as the `WorkingModel.ipynb` notebook.
4. Open the `WorkingModel.ipynb` notebook in Jupyter Notebook or JupyterLab.
5. Execute the code cells in the notebook sequentially to train the model and predict unique words.

## Model Architecture

The word prediction model is built using TensorFlow and Keras. It consists of the following layers:

1. Embedding Layer: Converts input words into dense vectors of fixed size.
2. LSTM Layer: Captures the sequential information from the embedded word vectors.
3. Dense Layer: Performs multiclass classification using a softmax activation function to predict the unique words.

The model is compiled with the `sparse_categorical_crossentropy` loss function and the Adam optimizer.

## Training

The training process involves the following steps:

1. Load the dataset from the JSON file.
2. Tokenize the sentences in the dataset using the `Tokenizer` class from Keras.
3. Pad the tokenized sequences to ensure they have the same length.
4. Create the model and compile it with the appropriate loss function and optimizer.
5. Prepare the word labels for training by using the next word as the label for each sequence.
6. Truncate the sequences and word labels to have the same length.
7. Train the model using the truncated sequences and word labels.

During training, the model will iterate over the dataset for a specified number of epochs to adjust its internal parameters and improve its performance. The training progress is displayed, showing the accuracy and loss at each epoch.

## Predicting Unique Words

After training the model, you can use it to predict unique words in unseen sentences. To make predictions, follow these steps:

1. Load the trained model using `tf.keras.models.load_model`.
2. Tokenize the input sentence using the same tokenizer used during training.
3. Pad the sequence to match the maximum length.
4. Make predictions using the trained model.
5. Interpret the predictions by applying a threshold to filter the predicted words based on their probabilities.
6. Display the predicted words.

## Dependencies

The following dependencies are required to run the code in `WorkingModel.ipynb`:

- TensorFlow
- Keras
- NumPy
- Jupyter Notebook

You can install the required dependencies by running `!pip install -r requirements.txt`.

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code for personal and commercial purposes. See the [LICENSE](LICENSE) file for more details.

---

This README provides an overview of the `WorkingModel.ipynb` notebook in the UniqueWorkingModel repository. For more detailed information, refer to the code and comments within the notebook file itself.

If you have any questions or need further assistance, feel free to reach out or create an issue in the GitHub repository.
