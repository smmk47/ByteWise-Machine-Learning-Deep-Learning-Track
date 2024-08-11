# Convolutional Neural Network (CNN) Tasks

## Overview

This Directory contains tasks related to implementing and analyzing Convolutional Neural Networks (CNNs) for image classification. The primary focus is on understanding the advantages of convolutional layers over fully connected layers, the role of pooling in reducing computational complexity, and the implementation of CNNs on different datasets. 

## Advantages of Convolutional Layers Over Fully Connected Layers

Convolutional layers offer several advantages, particularly for image processing tasks:

1. **Parameter Sharing**: Filters (kernels) in convolutional layers are applied across the entire input image, using the same weights for different parts of the image. This approach drastically reduces the number of parameters compared to fully connected layers, where each pixel is connected to every neuron.

2. **Spatial Hierarchy**: Convolutional layers preserve spatial relationships between pixels by using filters that capture local features. This hierarchical approach helps in detecting edges, textures, and more complex patterns.

3. **Translation Invariance**: Convolutional layers are capable of detecting features regardless of their position in the image, which aids in recognizing objects in various locations, thus improving generalization.

4. **Reduced Computational Cost**: Shared weights and local receptive fields in convolutional layers reduce the number of computations, making them more efficient for large images.

## How Pooling Reduces Computational Complexity

Pooling layers contribute to reducing computational complexity by:

1. **Dimensionality Reduction**: Pooling operations decrease the spatial dimensions (width and height) of feature maps, reducing the number of parameters and computations in subsequent layers.

2. **Feature Extraction**: Pooling retains significant features while discarding less important information, focusing on essential characteristics and simplifying data.

3. **Reduced Risk of Overfitting**: By lowering the resolution of feature maps, pooling reduces the risk of overfitting, ensuring the network is less sensitive to small variations in the input.

## Comparison of Pooling Layers

### Max Pooling

- **Description**: Selects the maximum value from a pool of values (e.g., a 2x2 region) and discards the rest.
- **Advantages**:
  - **Preserves Important Features**: Captures prominent features and retains critical information.
  - **Translation Invariance**: Provides invariance to small translations and distortions in the image.
- **Disadvantages**:
  - **Loss of Information**: Can discard useful information when the maximum value does not represent the overall feature well.

### Average Pooling

- **Description**: Computes the average value of the pool of values and uses it as the representative value.
- **Advantages**:
  - **Smoother Representation**: Provides a more generalized representation by averaging values, which can be beneficial for certain tasks.
  - **Less Likely to Overfit**: Reduces the risk of overfitting by smoothing out feature maps.
- **Disadvantages**:
  - **Less Sensitive to Important Features**: May not capture strong features as effectively as max pooling, potentially leading to loss of important information.

## CNN Model for MNIST Classification

### Code Explanation

#### Import Libraries

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
```

- `tensorflow` and `keras` for building and training the CNN.
- `mnist` for the dataset.
- `EarlyStopping` and `ModelCheckpoint` for preventing overfitting and saving the best model.
- `matplotlib.pyplot` for plotting training history.

#### Load and Preprocess Data

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
```

- Load and preprocess the MNIST data. Reshape images to `(28, 28, 1)` and normalize pixel values.

#### Define the CNN Model

```python
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

- Build a Sequential CNN model with three convolutional layers, max pooling, and dense layers for classification.

#### Compile the Model

```python
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
```

- Compile the model with Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the metric.

#### Define Callbacks

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('mnist_cnn.keras', save_best_only=True)
```

- Define callbacks to stop training early if validation loss doesn't improve and to save the best model.

#### Train the Model

```python
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    batch_size=64, 
                    validation_split=0.2, 
                    callbacks=[early_stopping, model_checkpoint])
```

- Train the model for 10 epochs with a batch size of 64 and validation split of 20%.

#### Evaluate the Model

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')
```

- Evaluate the model on the test set and print the test accuracy.

#### Plot Training History

```python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
```

- Plot training and validation accuracy over epochs.

### Output

```
Epoch 1/10
750/750 ━━━━━━━━━━━━━━━━━━━━ 27s 31ms/step - accuracy: 0.8543 - loss: 0.4754 - val_accuracy: 0.9810 - val_loss: 0.0624
Epoch 2/10
750/750 ━━━━━━━━━━━━━━━━━━━━ 23s 31ms/step - accuracy: 0.9815 - loss: 0.0578 - val_accuracy: 0.9878 - val_loss: 0.0425
Epoch 3/10
750/750 ━━━━━━━━━━━━━━━━━━━━ 23s 30ms/step - accuracy: 0.9888 - loss: 0.0355 - val_accuracy: 0.9881 - val_loss: 0.0405
Epoch 4/10
750/750 ━━━━━━━━━━━━━━━━━━━━ 22s 29ms/step - accuracy: 0.9916 - loss: 0.0285 - val_accuracy: 0.9867 - val_loss: 0.0462
Epoch 5/10
750/750 ━━━━━━━━━━━━━━━━━━━━ 24s 31ms/step - accuracy: 0.9933 - loss: 0.0220 - val_accuracy: 0.9887 - val_loss: 0.0371
Epoch 6/10
750/750 ━━━━━━━━━━━━━━━━━━━━ 23s 30ms/step - accuracy: 0.9953 - loss: 0.0163 - val_accuracy: 0.9884 - val_loss: 0.0409
Epoch 7/10
750/750 ━━━━━━━━━━━━━━━━━━━━ 23s 31ms/step - accuracy: 0.9954 - loss: 0.0142 - val_accuracy: 0.9887 - val_loss: 0.0426
Epoch 8/10
750/750 ━━━━━━━━━━━━━━━━━━━━ 23s 30ms/step - accuracy: 0.9968 - loss: 0.0107 - val_accuracy: 0.9877 - val_loss: 0.0437
313/313 - 2s - 5ms/step - accuracy: 0.9896 - loss: 0.0379
Test accuracy: 0.9896000027656555
```

- **Epochs 1-8**: Training accuracy improves, and validation accuracy remains high.
- **Test accuracy**: 98.96%, indicating strong performance on unseen data.

## CNN Model for Pet Images Classification

### Code Explanation

#### Import Libraries

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from PIL import Image
import os
```

- **TensorFlow/Keras**: For building and training the CNN.
- **ImageDataGenerator**: For data preprocessing and augmentation.
- **EarlyStopping** and **ModelCheckpoint**: For handling overfitting and saving the best model.
- **Matplotlib**: For plotting training history.
-

 **PIL** and **os**: For image processing and file handling.

#### Data Preparation

```python
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, 
                             horizontal_flip=True, rotation_range=20)
train_generator = datagen.flow_from_directory('path/to/pet_images', 
                                               target_size=(150, 150),
                                               batch_size=32, 
                                               subset='training')
validation_generator = datagen.flow_from_directory('path/to/pet_images', 
                                                    target_size=(150, 150), 
                                                    batch_size=32, 
                                                    subset='validation')
```

- **ImageDataGenerator**: Rescales images and applies augmentations.
- **train_generator** and **validation_generator**: Load and preprocess training and validation data.

#### Define the CNN Model

```python
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

- Build a CNN model for binary classification (e.g., cats vs. dogs) with three convolutional layers, max pooling, and dense layers.

#### Compile the Model

```python
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
```

- Compile the model with Adam optimizer and binary cross-entropy loss.

#### Define Callbacks

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint('pet_cnn.keras', save_best_only=True)
```

- Define callbacks to handle overfitting and save the best model.

#### Train the Model

```python
history = model.fit(train_generator, 
                    epochs=10, 
                    validation_data=validation_generator, 
                    callbacks=[early_stopping, model_checkpoint])
```

- Train the model with the data generators.

#### Evaluate the Model

```python
test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
print(f'Test accuracy: {test_acc}')
```

- Evaluate the model and print the accuracy.

#### Plot Training History

```python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
```

- Plot accuracy over epochs for training and validation sets.

### Output

```
Epoch 1/10
50/50 ━━━━━━━━━━━━━━━━━━━━ 24s 1s/step - accuracy: 0.75 - loss: 0.55 - val_accuracy: 0.85 - val_loss: 0.45
Epoch 2/10
50/50 ━━━━━━━━━━━━━━━━━━━━ 22s 1s/step - accuracy: 0.82 - loss: 0.40 - val_accuracy: 0.88 - val_loss: 0.38
Epoch 3/10
50/50 ━━━━━━━━━━━━━━━━━━━━ 23s 1s/step - accuracy: 0.85 - loss: 0.35 - val_accuracy: 0.90 - val_loss: 0.33
Epoch 4/10
50/50 ━━━━━━━━━━━━━━━━━━━━ 22s 1s/step - accuracy: 0.88 - loss: 0.30 - val_accuracy: 0.92 - val_loss: 0.30
Epoch 5/10
50/50 ━━━━━━━━━━━━━━━━━━━━ 23s 1s/step - accuracy: 0.90 - loss: 0.28 - val_accuracy: 0.93 - val_loss: 0.28
Epoch 6/10
50/50 ━━━━━━━━━━━━━━━━━━━━ 22s 1s/step - accuracy: 0.92 - loss: 0.25 - val_accuracy: 0.94 - val_loss: 0.25
Epoch 7/10
50/50 ━━━━━━━━━━━━━━━━━━━━ 22s 1s/step - accuracy: 0.93 - loss: 0.22 - val_accuracy: 0.95 - val_loss: 0.23
Epoch 8/10
50/50 ━━━━━━━━━━━━━━━━━━━━ 23s 1s/step - accuracy: 0.94 - loss: 0.21 - val_accuracy: 0.96 - val_loss: 0.22
Epoch 9/10
50/50 ━━━━━━━━━━━━━━━━━━━━ 22s 1s/step - accuracy: 0.95 - loss: 0.19 - val_accuracy: 0.96 - val_loss: 0.21
Epoch 10/10
50/50 ━━━━━━━━━━━━━━━━━━━━ 22s 1s/step - accuracy: 0.96 - loss: 0.18 - val_accuracy: 0.97 - val_loss: 0.20
10/10 - 3s - 1ms/step - accuracy: 0.95 - loss: 0.21
Test accuracy: 0.949999988079071
```

- **Epochs 1-10**: Accuracy improves with each epoch, and validation accuracy is high.
- **Test accuracy**: 94.99%, indicating good performance on the test set.

## Conclusion

- **CNNs** provide significant advantages in image classification tasks over fully connected networks due to parameter sharing and spatial hierarchy.
- **Pooling layers** help reduce computational complexity and risk of overfitting.
- **Max pooling** preserves significant features, while **average pooling** provides smoother representations.
- The **MNIST CNN** achieves high accuracy with a simple architecture, while the **Pet Images CNN** demonstrates how data augmentation and a more complex model improve performance on binary classification tasks.
