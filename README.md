#Hybrid AI: YOLOv8 + Custom Transfer Learning

### Results
- YOLOv8n detected: `cat: 0.90 confidence`
- Custom Model (Day61): `Class 0 (CAT) - 99.98% confidence`
- Inference: 685.5ms

### What is Hybrid?
YOLO finds WHERE the object is (bounding box).
Custom model says WHAT it is with high accuracy (99.98% vs YOLO's 90%).

### Files
- `project7.py` - Full code working in Jupyter (no Streamlit)
- `cat.jpg` - Test image
- `bus_output.jpg` - YOLO Day62 test

### How to run
```python
pip install ultralytics tensorflow
# open Day63_Jupyter.ipynb and run cells
