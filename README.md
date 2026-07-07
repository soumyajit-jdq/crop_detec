# 🌾 Crop Detection & Classification System

An end-to-end CNN-based crop detection and classification system that identifies **139 types of crops/plants** from images. Built with TensorFlow/Keras, served via FastAPI.

---

## 🏗️ Project Structure

```
crop_detection/
├── dataset/                     # Image dataset
│   ├── 140_crops_list.txt       # List of all crop classes
│   ├── RGB_224x224/             # Pre-processed 224×224 RGB images
│   │   └── RGB_224x224/
│   │       ├── train/           # Training images (139 class folders)
│   │       ├── val/             # Validation images
│   │       └── test/            # Test images
│   ├── BGR_224x224/             # BGR variant
│   ├── GRAY_224x224/            # Grayscale variant
│   └── Raw/                     # Original raw images
├── models/                      # Trained model files (generated)
│   ├── crop_cnn_*.h5            # Keras model
│   ├── crop_cnn.onnx            # ONNX export
│   ├── crop_cnn_saved_model/    # TF SavedModel
│   └── class_names.json         # Class label mapping
├── training/                    # Training module
│   ├── config/config.yaml       # Configuration
│   ├── src/                     # Source code
│   ├── api/                     # FastAPI server
│   ├── README.md                # Detailed training docs
│   ├── pyproject.toml           # Dependencies
│   └── requirements.txt
└── _boilerplate/                # Boilerplate templates
```

## 🚀 Quick Start

```bash
cd training

# 1. Create environment
python -m venv .venv
.venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle
# https://www.kaggle.com/datasets/omrathod2003/140-most-popular-crops-image-dataset

# 4. Train the model
python -m src.train

# 5. Evaluate
python -m src.evaluate

# 6. Export to ONNX
python -m src.export

# 7. Start API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 8. Predict via API
curl -X POST http://localhost:8000/api/v1/predict -F "file=@crop_image.jpg"
```

## 🔧 Model Architectures

Toggle between architectures in `config/config.yaml`:

| Architecture | Config Value | Description |
|---|---|---|
| Custom CNN | `model_type: "custom"` | 4-block deep CNN from scratch |
| EfficientNetB3 | `model_type: "transfer"`, `backbone: "efficientnetb3"` | Transfer learning (default) |
| ResNet50 | `model_type: "transfer"`, `backbone: "resnet50"` | Transfer learning |

## 📊 Dataset

**140 Most Popular Crops Image Dataset** — [Kaggle](https://www.kaggle.com/datasets/omrathod2003/140-most-popular-crops-image-dataset)

- 139 crop/plant classes
- 224×224 RGB images
- Pre-split: train / val / test

## 📖 Documentation

See [training/README.md](training/README.md) for detailed documentation on:
- Configuration options
- Training procedures
- Evaluation metrics
- Model export formats
- API endpoint usage

## License

MIT License
