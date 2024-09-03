# Sentiment Analysis on IMDb Movie Reviews: RNN and Hybrid Models

## Introduction
The objective of this project is to perform sentiment analysis on IMDb movie reviews using two deep learning models: a Recurrent Neural Network (RNN) and a hybrid model that combines a pre-trained GPT-2 transformer with Convolutional Neural Networks (CNN) and RNN layers. The models were trained to classify movie reviews as either positive or negative.

## Dataset
The dataset used is the IMDb Movie Reviews dataset, which consists of 50,000 movie reviews labeled as positive or negative.

## Preprocessing
### Data Cleaning
The text data is cleaned by removing non-alphabetic characters and converting all text to lowercase using the following function:

```python
def clean_text(text):  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    return text.lower()
```

### Label Encoding
The labels (sentiment) are encoded as 1 for positive reviews and 0 for negative reviews.

### Data Splitting
The dataset is split into training and testing sets with an 80-20 split.

## Tokenization
GPT-2's tokenizer is used to tokenize the reviews. This tokenizer encodes the text into token IDs and attention masks, which are then fed into the models.

```python
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
```

## Model Architectures

### 1. RNN Model
The RNN model consists of an embedding layer followed by an RNN layer and a fully connected layer for binary classification.

```python
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(1, x.size(0), 64).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
```

### 2. Hybrid Model (GPT-2 + CNN + RNN)
The hybrid model leverages the GPT-2 transformer model to extract features, which are then passed through a CNN layer and an RNN layer before final classification.

```python
class HybridModel(nn.Module):
    def __init__(self, transformer_model, hidden_size, num_classes):
        super(HybridModel, self).__init__()
        self.transformer = transformer_model
        self.cnn = nn.Conv1d(in_channels=768, out_channels=64, kernel_size=3, padding=1)
        self.rnn = nn.RNN(64, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        transformer_output = self.transformer(input_ids, attention_mask=attention_mask).last_hidden_state
        transformer_output = transformer_output.permute(0, 2, 1)
        cnn_output = self.cnn(transformer_output)
        cnn_output = cnn_output.permute(0, 2, 1)
        h0 = torch.zeros(1, cnn_output.size(0), 64).to(device)
        rnn_output, _ = self.rnn(cnn_output, h0)
        out = self.fc(rnn_output[:, -1, :])
        return out
```

## Training
Both models were trained for 5 epochs using the Binary Cross-Entropy loss function and the Adam optimizer.

### RNN Training
The RNN model was trained with the following loss progression:

- **Epoch 1:** Loss = 0.6781
- **Epoch 2:** Loss = 0.6881
- **Epoch 3:** Loss = 0.6373
- **Epoch 4:** Loss = 0.5908
- **Epoch 5:** Loss = 0.7011

### Hybrid Model Training
The hybrid model was trained with the following loss progression:

- **Epoch 1:** Loss = 0.7062
- **Epoch 2:** Loss = 0.6998
- **Epoch 3:** Loss = 0.6762
- **Epoch 4:** Loss = 0.6951
- **Epoch 5:** Loss = 0.6952

## Evaluation

### RNN Model Evaluation
The RNN model's performance was evaluated on the test set:

- **Accuracy:** 55%
- **Precision:** 0.54 (Negative), 0.55 (Positive)
- **Recall:** 0.55 (Negative), 0.54 (Positive)
- **F1-Score:** 0.55 (Negative), 0.54 (Positive)

The model shows a balanced precision and recall but with a relatively low accuracy of 55%.

### Hybrid Model Evaluation
The hybrid model's performance was notably worse:

- **Accuracy:** 50%
- **Precision:** 0.50 (Negative), 0.00 (Positive)
- **Recall:** 1.00 (Negative), 0.00 (Positive)
- **F1-Score:** 0.66 (Negative), 0.00 (Positive)

The hybrid model predicts all test samples as negative, resulting in an undefined precision and F1-score for the positive class.

## Testing on Sample Reviews
The models were also tested on several individual reviews to compare their predictions.

- **Positive Reviews:**
  - RNN correctly identified positive reviews as positive.
  - Hybrid Model incorrectly identified all positive reviews as negative.

- **Negative Reviews:**
  - Both models correctly identified negative reviews as negative, except for one case where the RNN incorrectly labeled a negative review as positive.

## Conclusion
- The RNN model showed moderate success in classifying IMDb movie reviews with an accuracy of 55%.
- The hybrid model, despite its complex architecture, underperformed with a 50% accuracy and an issue of class imbalance in predictions.
- Improvements such as hyperparameter tuning, model adjustments, or more sophisticated techniques (like attention mechanisms) may be necessary for better performance.

## Recommendations
Given the results, further tuning and experimentation with both models are recommended, especially for the hybrid model, which may have suffered from overfitting or issues related to the training process. Additionally, exploring other model architectures or combinations may yield better results in sentiment analysis tasks.
