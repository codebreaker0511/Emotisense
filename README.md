# Emotion Detector

Emotion Detector is a real-time facial emotion recognition system built using TensorFlow, Keras, and OpenCV. The model is trained on the FER-2013 dataset and predicts human emotions from live webcam input.

## Dataset

**Dataset:** FER-2013 (Facial Expression Recognition)

**Image Size:** 48 × 48 grayscale images

**Emotion Classes:**

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

## Features

* Real-time emotion detection using webcam input
* Face detection using OpenCV Haar Cascades
* Deep Convolutional Neural Network (CNN) for emotion classification
* Supports seven emotion categories
* Live emotion prediction overlay on video stream

## Project Structure

```text
Emotion_Detector(Week_6+7)
│
├── Emotion_Detector.ipynb
├── Emotion_Detector_Model.h5
├── webcam_detector.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Navigate to the project directory

```bash
cd Emotisense
cd "Emotion_Detector(Week_6+7)"
```

### 2. Create a virtual environment (recommended)

```bash
py -3.10 -m venv venv310
```

### 3. Activate the virtual environment

**Windows PowerShell**

```bash
.\venv310\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python webcam_detector.py
```

### 6. Exit

Press **Q** to close the webcam window.

## Model Performance

Final training results:

* Training Accuracy: **61.57%**
* Validation Accuracy: **62.64%**

The model achieves reasonable performance on FER-2013 while remaining lightweight enough for real-time inference.

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Jupyter Notebook

## Conclusion

The model successfully performs real-time emotion recognition from webcam input. The overall accuracy is limited by variations and imbalance within the FER-2013 dataset, where certain emotions have significantly more training examples than others. Despite these challenges, the model achieves over 60% validation accuracy and demonstrates practical real-time emotion classification.
