# TinyMLino

**TinyMLino.h** (`TinyMLP`) is a lightweight C++ library for microcontrollers (e.g., ESP32, Arduino) that implements a fully-connected neural network (Multi-Layer Perceptron) capable of running both on-device **training** and **inference** directly in RAM, with zero external dependencies.

## Key Features
* **Flexible Architecture:** Supports an arbitrary number of dense layers with **ReLU** activation in hidden layers and **Softmax + Cross-Entropy** at the output;
* **On-Device Training:** Trains via Stochastic Gradient Descent (**SGD**) one sample at a time (batch size = 1) using uniform Glorot weight initialization;
* **Automatic Input Scaling:** Accepts both `uint8_t` arrays (e.g., `0–255` pixel values, automatically scaled to `[0, 1]`) and `float` arrays (used as-is);
* **Zero Dependencies:** Relies solely on `<Arduino.h>` and `<math.h>`, dynamically allocating `float32` weights in RAM.

## Installation
1. Download or clone this library into your Arduino `libraries/` folder (or copy `TinyMLino.h` directly into your sketch folder);
2. Include the header in your project with `#include <TinyMLino.h>`.

## Basic Example (MNIST)

The following example demonstrates the full lifecycle of the network: defining the architecture (`784 -> 32 -> 10`), training on a dataset stored in flash memory, and running inference on a single sample.

```cpp
#include <TinyMLino.h>
#include "mnist_flash_dataset.h" 

TinyMLP net;

const float LEARNING_RATE = 0.03f;
const int   EPOCHS        = 10;

void setup() {
  Serial.begin(115200);
  delay(1000);

  // 1. Network declaration: 784 inputs -> 32 hidden neurons -> 10 classes
  const int layers[] = {784, 32, 10};
  if (!net.begin(layers, 3)) {
    Serial.println("Insufficient RAM to allocate the network!");
    return;
  }
  net.printSummary();

  // 2. Training: flash datasets are [N][784] matrices; &x[0][0] provides a contiguous pointer
  Serial.printf("Initial accuracy: %.2f%%\n",
                net.evaluate(&test_images[0][0], test_labels, NUM_TEST_SAMPLES));

  for (int epoch = 1; epoch <= EPOCHS; epoch++) {
    float loss = net.trainEpoch(&train_images[0][0], train_labels, NUM_TRAIN_SAMPLES, LEARNING_RATE);
    float acc  = net.evaluate(&test_images[0][0], test_labels, NUM_TEST_SAMPLES);
    Serial.printf("Epoch %2d/%d | Loss: %.4f | Test Accuracy: %.2f%%\n", epoch, EPOCHS, loss, acc);
  }

  // 3. Inference on a single image
  int predicted = net.predict(test_images[0]);
  Serial.printf("\nTest image 0: true class %d, predicted %d (confidence %.1f%%)\n",
                test_labels[0], predicted, net.probabilities()[predicted] * 100.0f);
}

void loop() {}
