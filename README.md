# AI & Machine Learning Engineering Lab 🔬

A curated collection of machine learning algorithms, deep learning architectures, computer vision experiments, and end-to-end applications developed from scratch and with modern frameworks.

---

## 📂 Repository Structure

```
.
├── classical-ml/                         # Classical ML models and tabular data analysis
│   ├── churn_rate.ipynb                  # Customer churn prediction
│   └── models-from-scratch/              # Pure Python & NumPy implementations
│       ├── DecisionTree.py
│       ├── random-forest.py
│       ├── shiv-knn.py
│       ├── shiv-log-reg.py
│       └── shivs-lr.py
├── deep-learning-fundamentals/           # Core DL theory & optimization
│   ├── backpropagation/                  # Multi-layer Perceptron & backprop from scratch
│   ├── optimization/                     # GD, EWMA, Xavier init, regularization, dropout, tuning
│   ├── pytorch-basics/                   # PyTorch tensors, autograd, custom training loops, audio classification
│   ├── rnn-nlp-basics/                   # BPTT, text preprocessing, and Word2Vec models
│   └── time-series-analysis/             # Time-series forecasting notebooks
├── computer-vision/                      # Vision models & OpenCV processing
│   ├── cnn-visualiser.ipynb              # Feature map activations & filter visualization
│   ├── the-simpson-saliency.ipynb        # Saliency maps & interpretability
│   ├── single-image-data-augmentation.ipynb
│   └── opencv/                           # Video capture, color spaces, transformations, H.264
├── projects/                             # End-to-end production applications
│   ├── brain-tumor-classification/       # Brain Tumor Detection Web App (Streamlit + PyTorch CNN)
│   └── volatility-aware-load-forecasting/# Deep learning time-series power consumption forecasting
├── datasets/                             # Tabular datasets & text transcripts
├── archives-and-installers/              # Environment binaries and scripts
├── 1301.3781v3.pdf                       # Word2Vec Research Paper (Mikolov et al.)
└── requirements.txt                      # Lab environment dependencies
```

---

## 🚀 Key Highlights

1. **Algorithms From Scratch:**
   - Decision Trees, Random Forests, K-Nearest Neighbors, Logistic Regression, Linear Regression, and Backpropagation implemented in pure NumPy.
2. **Deep Learning Optimization & Fundamentals:**
   - Mathematical exploration of Gradient Descent variants, Exponentially Weighted Moving Averages, Glorot/Xavier Initialization, Dropout mechanisms, and Hyperparameter Tuning with Keras Tuner & Optuna.
3. **Computer Vision & Saliency:**
   - CNN feature map visualizations and backprop saliency mapping for model interpretability.
4. **End-to-End Applications:**
   - **Brain Tumor Classification:** Full pipeline with MRI preprocessing (CLAHE, threshold contour cropping) and real-time Streamlit classification UI.
   - **Volatility-Aware Load Forecasting:** Hybrid deep learning architecture for multi-step energy load forecasting.

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone git@github.com:sshiv-1/ai-engineering-lab.git
cd ai-engineering-lab

# Install dependencies
pip install -r requirements.txt
```

---
*Maintained by Shiv.*
