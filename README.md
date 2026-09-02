# EmoVerse — Facial Emotion Detection

EmoVerse is a deep-learning based emotion detection project that recognizes
human facial emotions from images/video frames.

## 🎭 Emotions Detected

The model classifies 7 emotions:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## 🧠 Project Architecture

The project uses a Convolutional Neural Network (CNN) trained on the
FER2013 facial expression dataset.

Pipeline:

Camera / Image
     ↓
Face Detection
     ↓
Face Cropping & Preprocessing
     ↓
CNN Emotion Model
     ↓
7-Class Emotion Prediction

## 📊 Dataset

The project uses the FER2013 dataset.

The dataset contains approximately 35,887 facial images divided into
training and testing data.

The dataset itself is NOT included in this repository because of its size.

## 📈 Model Versions

Several versions of the emotion detection model were trained and evaluated.

| Model | Accuracy |
|-------|----------|
| Model V1 | 58.55% |
| Model V2 | 62.38% |
| Model V3 | 63.36% |

Model V3 currently provides the best performance.

## 🛠️ Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- FER2013 Dataset

## 📁 Project Structure

```text
EmoVerse/
│
├── inference/
│   └── emotion_camera.py
│
├── models/
│   └── trained model files (not uploaded)
│
├── evaluate_emotion.py
├── face_crop.py
├── face_detection.py
├── preprocess_face.py
├── test_camera.py
├── train_emotion.py
├── .gitignore
└── README.md
