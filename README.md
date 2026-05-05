# Face Detection — HOG Features + SVM Classifier

## Problem
Detect and localize human faces in images without deep learning — implementing a classical computer vision pipeline from feature extraction to sliding window detection.

## Why This Approach
HOG (Histogram of Oriented Gradients) + SVM is the foundation of pre-deep-learning object detection. Understanding it matters because:
1. It works in constrained environments with limited data
2. The feature engineering logic (gradient orientation histograms) transfers to other signal domains — vibration analysis, inertial sensor data, spectral features
3. The detection process is fully interpretable

## Pipeline

```
Raw Image
   ↓
HOG Feature Extraction  ← gradient magnitudes + orientation histograms in local cells
   ↓
SVM Classifier          ← trained on positive (face) and negative (background) patches
   ↓
Sliding Window          ← apply classifier at multiple scales across full image
   ↓
Non-Maximum Suppression ← merge overlapping detections
```

## Dataset
Labeled Faces in the Wild (LFW) — standard benchmark.  
Olivetti Faces used as additional positive samples.

## Key Learning
The main failure mode is background texture generating HOG patterns similar to facial gradients — false positives concentrated in high-contrast regions. Hard negative mining is the standard fix.

## What I Would Do Differently
- Add hard negative mining: collect false positives, retrain — standard technique to cut false positive rate significantly
- Implement image pyramid with integral images for efficiency

## Stack
`Python` · `scikit-learn` · `scikit-image` · `numpy` · `matplotlib`

---
*Part of [Davide Incaini's data science portfolio](https://github.com/davideincaini)*
