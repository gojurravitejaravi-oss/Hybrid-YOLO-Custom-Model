{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "09a659ce-5059-4f35-b219-b5d7f3d686d1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading YOLO...\n",
      "Loading your custom model...\n",
      "WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Both loaded!\n"
     ]
    }
   ],
   "source": [
    "from ultralytics import YOLO\n",
    "from tensorflow.keras.models import load_model\n",
    "\n",
    "print(\"Loading YOLO...\")\n",
    "yolo_model = YOLO(\"yolov8n.pt\")\n",
    "\n",
    "print(\"Loading your custom model...\")\n",
    "custom_model = load_model(\"model.h5\") # put model.h5 in same folder\n",
    "\n",
    "print(\"Both loaded!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1e2cd831-2727-4107-8529-6dcde907245c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "0: 640x640 1 cat, 685.5ms\n",
      "Speed: 43.3ms preprocess, 685.5ms inference, 42.6ms postprocess per image at shape (1, 3, 640, 640)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\raviteja\\AppData\\Local\\Temp\\ipykernel_10144\\400486683.py:17: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown\n",
      "  plt.show()\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "YOLO Details:\n",
      "- cat: 0.90\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 5s/step\n",
      "\n",
      "Custom Model Prediction:\n",
      "[[    0.99984  0.00016293]]\n",
      "Result: Class 0 - 99.98% confidence\n"
     ]
    }
   ],
   "source": [
    "from PIL import Image\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "img_path = \"cat.jpg\" \n",
    "img = Image.open(img_path).convert(\"RGB\")\n",
    "\n",
    "# 1. YOLO Detection\n",
    "results = yolo_model.predict(img)\n",
    "plotted = results[0].plot() \n",
    "\n",
    "plt.figure(figsize=(10,5))\n",
    "plt.imshow(plotted)\n",
    "plt.title(f\"YOLO Found {len(results[0].boxes)} objects\")\n",
    "plt.axis(\"off\")\n",
    "plt.show()\n",
    "\n",
    "print(\"YOLO Details:\")\n",
    "for box in results[0].boxes:\n",
    "    label = yolo_model.names[int(box.cls)]\n",
    "    conf = float(box.conf)\n",
    "    print(f\"- {label}: {conf:.2f}\")\n",
    "\n",
    "# 2. Your Custom Model\n",
    "img_resized = img.resize((224, 224))\n",
    "arr = np.expand_dims(np.array(img_resized)/255.0, axis=0)\n",
    "pred = custom_model.predict(arr)\n",
    "\n",
    "print(\"\\nCustom Model Prediction:\")\n",
    "print(pred)\n",
    "\n",
    "if pred.shape[1] == 1:\n",
    "    prob = float(pred[0][0])\n",
    "    label = \"DOG\" if prob > 0.5 else \"CAT\"\n",
    "    print(f\"Result: {label} - {prob:.2%} confidence\")\n",
    "else:\n",
    "    idx = np.argmax(pred)\n",
    "    print(f\"Result: Class {idx} - {np.max(pred):.2%} confidence\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f865e297-255f-4536-9655-028c8067aeb4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.21"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
