# Crop Detection & Classification — CNN Training Module

A complete CNN-based pipeline for classifying **139 crop/plant types** from images using TensorFlow / Keras. Supports both a **custom CNN** built from scratch and **transfer learning** (ResNet50 / EfficientNetB3) with a simple config toggle. Includes training, evaluation, model export (H5 / ONNX / SavedModel), and a **FastAPI inference server**.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Model Export](#model-export)
- [CLI Prediction](#cli-prediction)
- [API Server](#api-server)
- [API Endpoints](#api-endpoints)

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    config/config.yaml                 │
│          (model_type, backbone, hyperparams)          │
└────────────────────────┬─────────────────────────────┘
                         │
                         ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────────┐
│  Dataset    │──▸│  Model      │──▸│  Training        │
│  (tf.data)  │   │  Factory    │   │  (2-phase)       │
│             │   │             │   │                   │
│ • train     │   │ • custom    │   │ Phase 1: frozen   │
│ • val       │   │   CNN       │   │ Phase 2: fine-    │
│ • test      │   │ • transfer  │   │   tune (transfer) │
│ • augment   │   │   learning  │   │                   │
└─────────────┘   └─────────────┘   └────────┬──────────┘
                                             │
                  ┌──────────────────────────┤
                  ▼                          ▼
          ┌──────────────┐          ┌──────────────┐
          │  Evaluate    │          │  Export       │
          │              │          │              │
          │ • accuracy   │          │ • .h5        │
          │ • F1 score   │          │ • .onnx      │
          │ • confusion  │          │ • SavedModel │
          │   matrix     │          │ • .tflite    │
          └──────────────┘          └──────┬───────┘
                                           │
                                           ▼
                                   ┌──────────────┐
                                   │  FastAPI      │
                                   │  /predict     │
                                   │  /health      │
                                   │  /classes     │
                                   └──────────────┘
```

---

## Project Structure

```
training/
├── config/
│   └── config.yaml              # All hyperparameters & model toggle
├── src/
│   ├── __init__.py
│   ├── config.py                # Load & validate config.yaml
│   ├── dataset.py               # Data loading, augmentation, tf.data pipeline
│   ├── models/
│   │   ├── __init__.py          # Model factory (build + compile)
│   │   ├── custom_cnn.py        # 4-block custom CNN
│   │   └── transfer_cnn.py      # ResNet50 / EfficientNetB3 transfer learning
│   ├── train.py                 # Training orchestrator (2-phase)
│   ├── evaluate.py              # Metrics, confusion matrix, classification report
│   ├── export.py                # Export to ONNX / SavedModel / TFLite
│   ├── predict.py               # CLI single-image prediction
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Loguru-based logging
│       └── callbacks.py         # Custom Keras callbacks
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── routes.py                # /predict, /health, /classes endpoints
│   └── schemas.py               # Pydantic request/response models
├── notebooks/
│   └── exploration.ipynb        # Exploration notebook
├── outputs/                     # Training outputs (auto-generated)
│   ├── logs/                    # CSV logs, app logs
│   └── tensorboard/             # TensorBoard event files
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md                    # This file
```

---

## Prerequisites

- **Python** 3.10 – 3.12
- **pip** or **uv** for package management
- **GPU** (recommended): NVIDIA GPU with CUDA 11.8+ for TensorFlow GPU acceleration
- **Dataset**: ~2 GB (139 crop classes, 224×224 images)

---

## Setup

### 1. Create Virtual Environment

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

# Or using standard venv
python -m venv .venv
.venv\Scripts\activate         # Windows
```

### 2. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

### 3. Download Dataset

Download the [140 Most Popular Crops Image Dataset](https://www.kaggle.com/datasets/omrathod2003/140-most-popular-crops-image-dataset) from Kaggle and extract it to:

```
crop_detection/dataset/RGB_224x224/RGB_224x224/
├── train/           # 139 class subfolders
├── val/
└── test/
```

Each subfolder (e.g., `Tomatoes plant/`, `Rice (Paddy) plant/`) contains images for that class.

---

## Configuration

All hyperparameters are in **`config/config.yaml`**:

### Model Toggle

Switch between architectures by changing one line:

```yaml
# Custom CNN from scratch
model_type: "custom"

# OR Transfer Learning
model_type: "transfer"
backbone: "efficientnetb3"   # or "resnet50"
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | `"transfer"` | `"custom"` or `"transfer"` |
| `backbone` | `"efficientnetb3"` | `"resnet50"` or `"efficientnetb3"` |
| `image_size` | `224` | Input image dimensions |
| `batch_size` | `32` | Training batch size |
| `epochs` | `50` | Max training epochs |
| `learning_rate` | `0.001` | Initial learning rate |
| `fine_tune_layers` | `30` | Backbone layers to unfreeze (transfer only) |
| `early_stopping_patience` | `10` | Epochs to wait before stopping |

---

## Dataset

The project uses the **140 Most Popular Crops Image Dataset** from Kaggle:
- **139 crop/plant classes** (folder-per-class structure)
- **Pre-split** into train / val / test
- **Image size**: 224×224 RGB
- **Source**: [Kaggle](https://www.kaggle.com/datasets/omrathod2003/140-most-popular-crops-image-dataset)

Example classes: `Apples plant`, `Rice (Paddy) plant`, `Tomatoes plant`, `Wheat plant`, etc.

---

## Training

Run from the `training/` directory:

```bash
# Train with default config (transfer learning + EfficientNetB3)
python -m src.train

# Train with custom config
python -m src.train --config path/to/custom_config.yaml

# Train custom CNN (change config.yaml first)
# Set model_type: "custom" in config/config.yaml, then:
python -m src.train
```

### Training Phases (Transfer Learning)

1. **Phase 1 — Frozen Backbone**: Train only the classification head (fast convergence)
2. **Phase 2 — Fine-tuning**: Unfreeze top N backbone layers, train with lower LR

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir outputs/tensorboard

# Training logs
cat outputs/logs/training_phase1_initial.csv
```

### Output

After training completes:
```
models/
├── crop_cnn_transfer_efficientnetb3.h5       # Final model
├── crop_cnn_transfer_efficientnetb3_best.h5  # Best checkpoint
└── class_names.json                           # Class label mapping
```

---

## Evaluation

```bash
# Evaluate with default model
python -m src.evaluate

# Evaluate a specific model
python -m src.evaluate --model models/crop_cnn_custom.h5
```

### Output

```
outputs/
├── classification_report.json    # Per-class precision, recall, F1
└── confusion_matrix.png          # Visual confusion matrix
```

---

## Model Export

```bash
# Export to all configured formats (H5, ONNX, SavedModel)
python -m src.export

# Export a specific model
python -m src.export --model models/crop_cnn_transfer_efficientnetb3.h5
```

### Exported Files

```
models/
├── crop_cnn_transfer_efficientnetb3.h5    # Keras H5
├── crop_cnn.onnx                           # ONNX format
└── crop_cnn_saved_model/                   # TensorFlow SavedModel
```

---

## CLI Prediction

```bash
# Predict with default model
python -m src.predict --image path/to/crop_image.jpg

# Specify model and top-K
python -m src.predict --image photo.jpg --model models/crop_cnn_custom.h5 --top-k 10
```

### Example Output

```
==================================================
  Predicted Crop : Rice (Paddy) plant
  Confidence     : 94.32%

  Top-5 Predictions:
    1. Rice (Paddy) plant                          94.32% ████████████████████████████
    2. Wheat plant                                  2.15% █
    3. Barley plant                                 1.08%
    4. Millet plant                                 0.67%
    5. Oats plant                                   0.43%
==================================================
```

---

## API Server

### Start the Server

```bash
# Development (with auto-reload)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Interactive Docs

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the Swagger UI.

---

## API Endpoints

### `POST /api/v1/predict` — Classify a crop image

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "file=@crop_image.jpg"
```

**Response:**
```json
{
  "predicted_class": "Rice (Paddy) plant",
  "confidence": 0.9432,
  "top_5": [
    {"class_name": "Rice (Paddy) plant", "confidence": 0.9432},
    {"class_name": "Wheat plant", "confidence": 0.0215},
    {"class_name": "Barley plant", "confidence": 0.0108},
    {"class_name": "Millet plant", "confidence": 0.0067},
    {"class_name": "Oats plant", "confidence": 0.0043}
  ]
}
```

### `GET /api/v1/health` — Health check

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 139,
  "model_type": "transfer"
}
```

### `GET /api/v1/classes` — List all crop classes

```bash
curl http://localhost:8000/api/v1/classes
```

```json
{
  "num_classes": 139,
  "classes": ["Aji pepper plant", "Almonds plant", "..."]
}
```

---

## License

This project is licensed under the MIT License.