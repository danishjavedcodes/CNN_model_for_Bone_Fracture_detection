# X-Ray Bone Fracture Detection with CNN (TensorFlow & Gradio)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-yellow.svg)](https://gradio.app/)

**Binary classification of musculoskeletal X-ray images as fractured or not fractured** using a lightweight **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**, plus an interactive **Gradio** web demo for upload-and-predict workflows.

> Educational and research use only. This project is **not** a medical device and must **not** be used for clinical diagnosis without validation by qualified professionals.

---

## Table of contents

- [Why this project](#why-this-project)
- [Features](#features)
- [How it works](#how-it-works)
- [Model architecture](#model-architecture)
- [Dataset](#dataset)
- [Quick start](#quick-start)
- [Train the CNN](#train-the-cnn)
- [Run the Gradio demo](#run-the-gradio-demo)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)
- [Related reading](#related-reading)
- [Author](#author)
- [Keywords](#keywords)

---

## Why this project

Bone fractures are among the most common musculoskeletal injuries worldwide. Radiologists review large volumes of **X-ray radiographs** every day; subtle fracture lines can be easy to miss under time pressure. **Deep learning** and **computer vision** can support screening by flagging likely positive cases for human review.

This repository provides a **minimal, readable CNN pipeline** for:

- **X-ray bone fracture detection**
- **Medical image classification** (binary: fractured vs. not fractured)
- **TensorFlow/Keras** training with augmentation
- **Gradio** deployment for quick inference demos

It is aimed at students, ML engineers, and researchers learning **CNN for medical imaging**, not at replacing radiologist judgment.

---

## Features

| Feature | Description |
|--------|-------------|
| **Custom CNN** | Small Conv2D → pooling → dense stack for fast training on modest hardware |
| **Data augmentation** | Rescale, shear, zoom, and horizontal flip via `ImageDataGenerator` |
| **TensorBoard logging** | Training metrics and histograms under `logs/fit/` |
| **Gradio UI** | Upload an X-ray image and get a text label in the browser |
| **Jupyter notebook** | End-to-end workflow in `x_ray_bone_fracture_detection.ipynb` |

---

## How it works

1. **Prepare data** — Organize images into folder-per-class layout (`fractured/` / `not_fractured/` or equivalent) for `flow_from_directory`.
2. **Preprocess** — Resize to **150×150**, normalize pixels to `[0, 1]`, apply augmentation on the training set.
3. **Train** — Fit the CNN with binary cross-entropy and track accuracy.
4. **Export** — Save weights (e.g. `x_ray.keras`) for inference.
5. **Deploy** — Load the model in Gradio and classify new uploads.

```text
X-ray image → resize (150×150) → normalize → CNN → sigmoid → "fractured" / "NOT fractured"
```

---

## Model architecture

| Layer | Configuration |
|-------|----------------|
| Input | `(150, 150, 3)` RGB |
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dense | 128 units, ReLU |
| Output | 1 unit, sigmoid (binary) |

- **Optimizer:** Adam  
- **Loss:** `binary_crossentropy`  
- **Default training:** `batch_size=32`, `epochs=10` (tune on your hardware and dataset)

---

## Dataset

The notebook references a curated X-ray dataset distributed via MEGA:

**Download:** [MEGA dataset link](https://mega.nz/file/zcdywLhI#fck4ufXy_o_Uiu0vGqh-cZiKHw5Xe_n4M2qWUWSheAI)

After download, extract and point `train_data_dir` and `test_data_dir` in the notebook to your **train** and **validation** folders.

**Expected directory layout:**

```text
dataset/
├── train/
│   ├── fractured/
│   └── not_fractured/
└── val/
    ├── fractured/
    └── not_fractured/
```

*(Class folder names must match what `flow_from_directory` discovers; rename if your archive uses different labels.)*

---

## Quick start

### Prerequisites

- Python 3.8+
- pip

### Install dependencies

```bash
pip install tensorflow numpy gradio keras
```

For GPU training, install TensorFlow with CUDA support following the [official TensorFlow install guide](https://www.tensorflow.org/install).

### Clone the repository

```bash
git clone https://github.com/danishjavedcodes/CNN_model_for_Bone_Fracture_detection.git
cd CNN_model_for_Bone_Fracture_detection
```

### Open the notebook

```bash
jupyter notebook x_ray_bone_fracture_detection.ipynb
```

Update `train_data_dir` and `test_data_dir`, then run all cells.

---

## Train the CNN

Core training steps (see the notebook for full code):

1. Configure paths to train/validation directories.
2. Build generators with `ImageDataGenerator` (`class_mode='binary'`).
3. Compile and `model.fit(train_generator, ...)`.
4. Monitor runs with TensorBoard:

```bash
tensorboard --logdir logs/fit
```

5. Save the trained model:

```python
model.save("x_ray.keras")
```

---

## Run the Gradio demo

After training (or once `x_ray.keras` is available in the project root):

1. Run the deployment cells in `x_ray_bone_fracture_detection.ipynb`, or reuse:

```python
import keras
import numpy as np
import gradio as gr

model = keras.models.load_model("x_ray.keras")

def classify_image(img):
    img = img.resize((150, 150))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    prediction = model.predict(arr)
    return "NOT fractured" if prediction > 0.5 else "fractured"

gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload an X-ray image"),
    outputs="text",
    title="Bone Fracture Classification",
    description="Upload an X-ray image; the model classifies it as fractured or not.",
).launch()
```

2. Open the local URL Gradio prints (typically `http://127.0.0.1:7860`).
3. Upload a test X-ray and read the prediction.

---

## Project structure

```text
CNN_model_for_Bone_Fracture_detection/
├── README.md                          # This file
├── x_ray_bone_fracture_detection.ipynb  # Train + Gradio workflow
├── x_ray.keras                        # Saved model (add after training)
├── logs/fit/                          # TensorBoard logs (created at train time)
└── .vscode/                           # Editor settings (optional)
```

---

## Tech stack

- [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/) — model building and training  
- [NumPy](https://numpy.org/) — array operations  
- [Gradio](https://gradio.app/) — web UI for inference  
- [Jupyter](https://jupyter.org/) — interactive development  
- [TensorBoard](https://www.tensorflow.org/tensorboard) — experiment tracking  

---

## Related reading

**In-depth walkthrough (same author, step-by-step CNN + Gradio tutorial):**

- [X-ray Bone Fracture Detection: Enhancing Medical Diagnosis with Convolutional Neural Networks (CNN)](https://pub.towardsai.net/x-ray-bone-fracture-detection-enhancing-medical-diagnosis-with-convolutional-neural-networks-cnn-d9e9f962ba22) — Danish Javed, *Towards AI*

**More articles by the author:**

- [A Beginner’s Guide to Predict Stock Prices with AI](https://medium.com/@danish_javed/a-beginners-guide-to-predicting-stock-prices-with-ai-by-danish-ef6ae482b1e9) — time-series / LSTM tutorial (separate project)

---

## Author

**Danish Javed** — AI/ML Engineer  

- GitHub: [@danishjavedcodes](https://github.com/danishjavedcodes)  
- Medium: [@danish_javed](https://medium.com/@danish_javed)  

If this repo helped you, consider starring it on GitHub and sharing the [tutorial article](https://pub.towardsai.net/x-ray-bone-fracture-detection-enhancing-medical-diagnosis-with-convolutional-neural-networks-cnn-d9e9f962ba22).

---

## Keywords

`bone fracture detection` · `X-ray classification` · `medical image deep learning` · `convolutional neural network` · `CNN TensorFlow` · `radiograph AI` · `musculoskeletal imaging` · `binary image classifier` · `Gradio medical demo` · `keras fracture model` · `computer vision healthcare` · `automated fracture screening`

---

## Disclaimer

This software is provided for **education and research** only. Predictions may be wrong; datasets and models can reflect bias and limited generalization. **Do not use this tool for medical decisions.** Always consult licensed healthcare providers for diagnosis and treatment.

## License

Unless otherwise noted, code in this repository is open source. Add a `LICENSE` file if you intend to specify terms (e.g. MIT).
