# IMDB Sentiment Analysis with Simple RNN and Hybrid RNN+CNN Models

## Overview

This project demonstrates sentiment analysis on the IMDB movie reviews dataset using two deep learning models:

1. **Simple RNN (Recurrent Neural Network)**
2. **Hybrid RNN+CNN (Convolutional Neural Network + Recurrent Neural Network)**

The models are trained on the IMDB dataset and evaluated on their ability to classify movie reviews as positive or negative. Additionally, five hard-coded reviews are tested to verify the models' predictions.

## Project Structure

- **IMDB Dataset**: The dataset consists of 50,000 movie reviews, with 25,000 for training and 25,000 for testing. Each review is labeled as either positive (1) or negative (0).
- **Simple RNN Model**: A basic RNN model with an embedding layer and a SimpleRNN layer.
- **Hybrid RNN+CNN Model**: A combination of CNN and RNN that leverages both convolutional and recurrent layers to capture different features of the reviews.

## Requirements

To run this project, you need the following dependencies:

- Python 3.6+
- TensorFlow 2.x
- NumPy

Install the required packages using pip:

```bash
pip install tensorflow numpy
```

## Code Explanation

### 1. **Loading and Preparing the Data**

The IMDB dataset is loaded using TensorFlow's `imdb` module. The data is preprocessed as follows:

- The reviews are limited to the top 10,000 most frequent words.
- Each review is padded to a fixed length of 500 words.

```python
max_features = 10000  # Number of words to consider as features
maxlen = 500  # Cut texts after this number of words

# Load the IMDB dataset
print("Loading data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# Pad the data
print("Pad sequences (samples x time)")
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
```

### 2. **Hardcoded Reviews**

Five hard-coded reviews are provided for testing the models. These reviews are encoded using the IMDB word index and padded to match the input shape expected by the models.

```python
# Hard-coded reviews
hardcoded_reviews = [
    "This movie was absolutely fantastic! The performances were incredible and the story was captivating.",
    "I really did not like this movie. It was too slow and the plot was very predictable.",
    "The film had some good moments, but overall it was just okay. Nothing too special.",
    "An amazing experience! I would definitely watch it again. Highly recommended!",
    "Terrible movie. The acting was bad, and the story made no sense."
]

# Encode and pad the reviews
encoded_reviews = [encode_review(review) for review in hardcoded_reviews]
padded_reviews = pad_sequences(encoded_reviews, maxlen=maxlen)
```

### 3. **Building and Training the Models**

Two models are defined and trained:

- **Simple RNN Model**: Comprises an embedding layer followed by a SimpleRNN layer and a dense layer with sigmoid activation.

```python
model_rnn = Sequential()
model_rnn.add(Embedding(max_features, 32))
model_rnn.add(SimpleRNN(32))
model_rnn.add(Dense(1, activation='sigmoid'))

model_rnn.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history_rnn = model_rnn.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
```

- **Hybrid RNN+CNN Model**: Includes an embedding layer, a convolutional layer with max pooling, followed by a SimpleRNN layer and a dense layer.

```python
model_hybrid = Sequential()
model_hybrid.add(Embedding(max_features, 32))
model_hybrid.add(Conv1D(32, 7, activation='relu'))
model_hybrid.add(MaxPooling1D(5))
model_hybrid.add(SimpleRNN(32))
model_hybrid.add(Dense(1, activation='sigmoid'))

model_hybrid.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history_hybrid = model_hybrid.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
```

### 4. **Evaluating the Models**

After training, the models are evaluated on the test set, and their accuracy and loss are printed.

```python
results_rnn = model_rnn.evaluate(x_test, y_test)
print(f"Test Loss, Test Accuracy for Simple RNN model: {results_rnn}")

results_hybrid = model_hybrid.evaluate(x_test, y_test)
print(f"Test Loss, Test Accuracy for Hybrid RNN+CNN model: {results_hybrid}")
```

### 5. **Testing Hardcoded Reviews**

The models predict the sentiment of each hardcoded review. The predictions are printed as "Positive" or "Negative" based on the probability score.

```python
for i, review in enumerate(hardcoded_reviews):
    test_review = padded_reviews[i]

    rnn_prediction = model_rnn.predict(np.array([test_review]))
    hybrid_prediction = model_hybrid.predict(np.array([test_review]))

    print(f"\nReview {i + 1}: {review}")
    print("Simple RNN model prediction:", "Positive" if rnn_prediction[0][0] > 0.5 else "Negative", f"({rnn_prediction[0][0]})")
    print("Hybrid RNN+CNN model prediction:", "Positive" if hybrid_prediction[0][0] > 0.5 else "Negative", f"({hybrid_prediction[0][0]})")
```

## Output

### Model Training and Evaluation

During training, you will see output for each epoch, showing the accuracy and loss. After training, the models are evaluated on the test set:

```plaintext
Evaluating Simple RNN model...
Test Loss, Test Accuracy for Simple RNN model: [0.6496554017066956, 0.8129600286483765]

Evaluating Hybrid RNN+CNN model...
Test Loss, Test Accuracy for Hybrid RNN+CNN model: [0.5545594692230225, 0.8575599789619446]

Summary of Model Performances:
Simple RNN model - Loss: 0.6496554017066956, Accuracy: 0.8129600286483765
Hybrid RNN+CNN model - Loss: 0.5545594692230225, Accuracy: 0.8575599789619446
```

### Hardcoded Review Predictions

The models make predictions on the hardcoded reviews. The output might look like this:

```plaintext
Review 1: This movie was absolutely fantastic! The performances were incredible and the story was captivating.
Simple RNN model prediction: Negative (0.07313903421163559)
Hybrid RNN+CNN model prediction: Negative (0.022225648164749146)

Review 2: I really did not like this movie. It was too slow and the plot was very predictable.
Simple RNN model prediction: Positive (0.8945593237876892)
Hybrid RNN+CNN model prediction: Negative (0.23435235023498535)

Review 3: The film had some good moments, but overall it was just okay. Nothing too special.
Simple RNN model prediction: Positive (0.9105430841445923)
Hybrid RNN+CNN model prediction: Positive (0.6295701265335083)

Review 4: An amazing experience! I would definitely watch it again. Highly recommended!
Simple RNN model prediction: Positive (0.8721972107887268)
Hybrid RNN+CNN model prediction: Negative (0.3438515365123749)

Review 5: Terrible movie. The acting was bad, and the story made no sense.
Simple RNN model prediction: Negative (0.020515333861112595)
Hybrid RNN+CNN model prediction: Negative (0.09388039261102676)
```

## Notes

- The Simple RNN model performs well on training and test data, but it can sometimes give unexpected results on hardcoded reviews due to overfitting.
- The Hybrid RNN+CNN model generally performs better on the test set and provides more consistent predictions, though it may also struggle with some reviews.

## Conclusion

This project demonstrates the use of RNN and hybrid RNN+CNN models for sentiment analysis. The models are trained on the IMDB dataset and evaluated for accuracy. The predictions on hardcoded reviews help in understanding the strengths and limitations of each model.

