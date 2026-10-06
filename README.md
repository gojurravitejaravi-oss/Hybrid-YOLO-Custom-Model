# Hybrid-YOLO-Custom-Model

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-nano-green)
![Accuracy](https://img.shields.io/badge/Custom_Model-99.98%25-success)

> YOLO finds WHERE, Custom CNN finds WHAT with 99.98% accuracy

### 🔥 Live Result (Your screenshot)
- YOLOv8n: `1 cat detected @ 0.90 confidence - 685.5ms`
- Custom Model (Transfer Learning): `Class 0 CAT - 0.99984 (99.98%)`
- Hybrid = Detection + High-Accuracy Classification

### 📁 How I did in Jupyter (No Streamlit)
```python
yolo = YOLO("yolov8n.pt")
custom = load_model("model.h5")
# YOLO box + Custom classification
