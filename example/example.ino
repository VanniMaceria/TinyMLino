#include <TinyMLino.h>
#include "mnist_flash_dataset.h" 

TinyMLP net;

const float LEARNING_RATE = 0.03f;
const int   EPOCHS        = 10;

void setup() {
  Serial.begin(115200);
  delay(1000);

  // 1. Dichiarazione della rete: 784 ingressi -> 32 neuroni nascosti -> 10 classi
  const int layers[] = {784, 32, 10};
  if (!net.begin(layers, 3)) {
    Serial.println("RAM insufficiente per creare la rete!");
    return;
  }
  net.printSummary();

  // 2. Training: i dataset in flash sono matrici [N][784], &x[0][0] ne dà il puntatore contiguo
  Serial.printf("Accuracy iniziale: %.2f%%\n",
                net.evaluate(&test_images[0][0], test_labels, NUM_TEST_SAMPLES));

  for (int epoch = 1; epoch <= EPOCHS; epoch++) {
    float loss = net.trainEpoch(&train_images[0][0], train_labels, NUM_TRAIN_SAMPLES, LEARNING_RATE);
    float acc  = net.evaluate(&test_images[0][0], test_labels, NUM_TEST_SAMPLES);
    Serial.printf("Epoca %2d/%d | Loss: %.4f | Test Accuracy: %.2f%%\n", epoch, EPOCHS, loss, acc);
  }

  // 3. Inferenza su una singola immagine
  int predetta = net.predict(test_images[0]);
  Serial.printf("\nImmagine di test 0: classe reale %d, predetta %d (confidenza %.1f%%)\n",
                test_labels[0], predetta, net.probabilities()[predetta] * 100.0f);
}

void loop() {}
