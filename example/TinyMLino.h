#ifndef TINY_MLP_H
#define TINY_MLP_H

#include <Arduino.h>
#include <math.h>

// ============================================================================
// TinyMLP: piccola rete neurale fully-connected con training e inferenza.
//
//   - Strati densi con ReLU nei livelli nascosti, Softmax + Cross-Entropy in uscita
//   - Training con SGD, un campione alla volta (batch = 1)
//   - Pesi float32 allocati in RAM, nessuna compressione del modello
//
// Uso minimo:
//     TinyMLP net;
//     const int layers[] = {784, 32, 10};   // ingresso, nascosti..., uscita
//     net.begin(layers, 3);
//     net.trainEpoch(images, labels, N, 0.03f);
//     int classe = net.predict(image);
//
// Gli input possono essere uint8_t (pixel 0-255, scalati automaticamente in [0,1])
// oppure float (usati così come sono).
// ============================================================================
class TinyMLP {
 public:
  TinyMLP() {}  //costruttore vuoto
  ~TinyMLP() { freeMemory(); }  //distruttore, viene chiamato in automatico
  //operazioni per vietare la copia dell'oggetto:
  /*
    Il motivo è che la classe contiene puntatori grezzi (W_, b_, a_, ...). Una copia predefinita duplicherebbe solo i puntatori, non i dati. 
    Si avrebbero due oggetti che puntano agli stessi buffer, e quando entrambi vengono distrutti la stessa memoria verrebbe liberata due volte (double free), 
    con un crash difficile da diagnosticare. Con = delete il compilatore ti blocca subito:
  */  
  TinyMLP(const TinyMLP&) = delete;
  TinyMLP& operator=(const TinyMLP&) = delete;

  // --------------------------------------------------------------------------
  // DICHIARAZIONE DELLA RETE
  // layerSizes: numero di neuroni di ogni livello, ingresso compreso
  // numSizes:   quanti elementi ha layerSizes (ex. {784, 32, 10} -> 3)
  // --------------------------------------------------------------------------
  bool begin(const int* layerSizes, int numSizes, unsigned long seed = 42) {
    freeMemory();
    if (numSizes < 2) return false;

    numLayers_ = numSizes - 1;  // numero di strati di pesi, ex. {784 X 32}, {32 X 10}
    sizes_ = (int*) malloc(numSizes * sizeof(int)); //array che punta al numero di neuroni per strato
    W_ = (float**) calloc(numLayers_, sizeof(float*));  // pesi, ex a 3 strati -> W[0] = {783 X 32}, W[1] = {32 X 10}
    b_ = (float**) calloc(numLayers_, sizeof(float*));  // bias
    a_ = (float**) calloc(numLayers_, sizeof(float*));  // attivazioni
    d_ = (float**) calloc(numLayers_, sizeof(float*));  // errori backward
    if (!sizes_ || !W_ || !b_ || !a_ || !d_) { freeMemory(); return false; }
    for (int i = 0; i < numSizes; i++) sizes_[i] = layerSizes[i];

    x_ = (float*) calloc(sizes_[0], sizeof(float));
    if (!x_) { freeMemory(); return false; }

    randomSeed(seed);
    for (int l = 0; l < numLayers_; l++) {
      int nIn = sizes_[l], nOut = sizes_[l + 1];
      W_[l] = (float*) malloc(nIn * nOut * sizeof(float));
      b_[l] = (float*) calloc(nOut, sizeof(float));
      a_[l] = (float*) calloc(nOut, sizeof(float));
      d_[l] = (float*) calloc(nOut, sizeof(float));
      if (!W_[l] || !b_[l] || !a_[l] || !d_[l]) { freeMemory(); return false; }

      // Inizializzazione Glorot uniforme: pesi in [-limite, +limite]
      float limit = sqrtf(6.0f / (nIn + nOut));
      for (int i = 0; i < nIn * nOut; i++) W_[l][i] = (randomUnit() * 2.0f - 1.0f) * limit;
    }
    return true;
  }

  // --------------------------------------------------------------------------
  // INFERENZA
  // --------------------------------------------------------------------------
  // Restituisce la classe con probabilità maggiore
  template <typename T>
  int predict(const T* input) {
    setInput(input);
    forward();
    const float* p = a_[numLayers_ - 1];
    int best = 0;
    for (int k = 1; k < outputSize(); k++) if (p[k] > p[best]) best = k;
    return best;
  }

  // Probabilità dell'ultima predict() / trainSample() (outputSize() valori)
  const float* probabilities() const { return a_[numLayers_ - 1]; }

  // --------------------------------------------------------------------------
  // TRAINING
  // --------------------------------------------------------------------------
  // Un passo di SGD su un campione. Restituisce la loss (cross-entropy).
  template <typename T>
  float trainSample(const T* input, int label, float learningRate) {
    setInput(input);
    forward();
    float loss = -logf(a_[numLayers_ - 1][label] + 1e-7f);
    backward(label, learningRate);
    return loss;
  }

  // Un'epoca su un dataset contiguo (count campioni di inputSize() valori ciascuno).
  // Restituisce la loss media.
  template <typename T>
  float trainEpoch(const T* inputs, const uint8_t* labels, int count, float learningRate) {
    float totalLoss = 0.0f;
    for (int n = 0; n < count; n++)
      totalLoss += trainSample(inputs + (size_t)n * inputSize(), labels[n], learningRate);
    return totalLoss / count;
  }

  // Accuracy (%) su un dataset
  template <typename T>
  float evaluate(const T* inputs, const uint8_t* labels, int count) {
    int correct = 0;
    for (int n = 0; n < count; n++)
      if (predict(inputs + (size_t)n * inputSize()) == labels[n]) correct++;
    return 100.0f * correct / count;
  }

  // --------------------------------------------------------------------------
  // INFORMAZIONI
  // --------------------------------------------------------------------------
  int inputSize()  const { return sizes_[0]; }
  int outputSize() const { return sizes_[numLayers_]; }

  int paramCount() const {
    int total = 0;
    for (int l = 0; l < numLayers_; l++) total += sizes_[l] * sizes_[l + 1] + sizes_[l + 1];
    return total;
  }

  void printSummary() const {
    Serial.println("=== TinyMLP ===");
    for (int l = 0; l < numLayers_; l++)
      Serial.printf("Layer %d: %d -> %d (%s)\n", l + 1, sizes_[l], sizes_[l + 1],
                    l < numLayers_ - 1 ? "ReLU" : "Softmax");
    Serial.printf("Parametri: %d (%d byte di RAM)\n", paramCount(), paramCount() * (int)sizeof(float));
  }

 private:
  int numLayers_ = 0;     // numero di strati di pesi
  int* sizes_ = nullptr;  // neuroni per livello (ingresso compreso)
  float* x_ = nullptr;    // copia dell'input corrente
  float** W_ = nullptr;   // W_[l][j * nIn + i]: peso dall'ingresso i al neurone j
  float** b_ = nullptr;   // bias
  float** a_ = nullptr;   // attivazioni (nell'ultimo strato: probabilità)
  float** d_ = nullptr;   // errori (delta) usati nel backward

  static float randomUnit() { return random(0, 10001) / 10000.0f; }  // [0, 1]

  // Pixel uint8 -> [0,1]; float invariato
  static float scaleFor(uint8_t) { return 1.0f / 255.0f; }
  static float scaleFor(float)   { return 1.0f; }

  template <typename T>
  void setInput(const T* input) {
    float s = scaleFor(T());
    for (int i = 0; i < sizes_[0]; i++) x_[i] = input[i] * s;
  }

  // Ingresso del livello l: l'input della rete oppure l'uscita del livello precedente
  const float* layerInput(int l) const { return l == 0 ? x_ : a_[l - 1]; }

  void forward() {
    for (int l = 0; l < numLayers_; l++) {
      int nIn = sizes_[l], nOut = sizes_[l + 1];
      const float* in = layerInput(l);
      for (int j = 0; j < nOut; j++) {
        const float* w = W_[l] + j * nIn;
        float sum = b_[l][j];
        for (int i = 0; i < nIn; i++) sum += w[i] * in[i];
        a_[l][j] = (l < numLayers_ - 1 && sum < 0.0f) ? 0.0f : sum;  // ReLU sui nascosti
      }
    }
    softmax(a_[numLayers_ - 1], outputSize());
  }

  static void softmax(float* v, int n) {
    float maxVal = v[0];
    for (int i = 1; i < n; i++) if (v[i] > maxVal) maxVal = v[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { v[i] = expf(v[i] - maxVal); sum += v[i]; }
    for (int i = 0; i < n; i++) v[i] /= sum;
  }

  // Retropropagazione dell'errore + aggiornamento SGD, dall'ultimo strato al primo
  void backward(int label, float lr) {
    int last = numLayers_ - 1;

    // Errore in uscita (Softmax + Cross-Entropy): probabilità - one-hot
    for (int k = 0; k < outputSize(); k++) d_[last][k] = a_[last][k] - (k == label ? 1.0f : 0.0f);

    for (int l = last; l >= 0; l--) {
      int nIn = sizes_[l], nOut = sizes_[l + 1];
      const float* in = layerInput(l);
      float* prevDelta = (l > 0) ? d_[l - 1] : nullptr;
      if (prevDelta) for (int i = 0; i < nIn; i++) prevDelta[i] = 0.0f;

      for (int j = 0; j < nOut; j++) {
        float* w = W_[l] + j * nIn;
        float dj = d_[l][j];
        for (int i = 0; i < nIn; i++) {
          if (prevDelta) prevDelta[i] += w[i] * dj;  // errore verso il livello precedente (peso PRIMA dell'update)
          w[i] -= lr * dj * in[i];                   // SGD sul peso
        }
        b_[l][j] -= lr * dj;                         // SGD sul bias
      }

      // Derivata della ReLU del livello precedente: l'errore passa solo dove l'attivazione è > 0
      if (prevDelta) for (int i = 0; i < nIn; i++) if (in[i] <= 0.0f) prevDelta[i] = 0.0f;
    }
  }

  void freeMemory() {
    if (W_) for (int l = 0; l < numLayers_; l++) { free(W_[l]); free(b_[l]); free(a_[l]); free(d_[l]); }
    free(W_); free(b_); free(a_); free(d_); free(x_); free(sizes_);
    W_ = b_ = a_ = d_ = nullptr; x_ = nullptr; sizes_ = nullptr; numLayers_ = 0;
  }
};

#endif
