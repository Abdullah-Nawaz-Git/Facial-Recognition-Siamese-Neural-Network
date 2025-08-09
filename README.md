# Automated Attendance System Using Facial Recognition

This repository contains a notebook that trains a Siamese Neural Network (SNN) for facial recognition, aiming to develop an automated attendance system. The project explores deep learning techniques for face verification and demonstrates a practical approach to attendance tracking using facial recognition.

---

## Project Overview

Traditional attendance methods are often inefficient, error-prone, and susceptible to cheating. This project investigates facial recognition as a solution to automate attendance in classrooms and institutions.

- **Initial Approach:** Training a Siamese Neural Network from scratch using a custom dataset combined with Labeled Faces in the Wild (LFW).  
- **Challenges:** Despite enhancements like using a pre-trained ResNet50 backbone and advanced training techniques, the model struggled to generalize to unseen faces in real-world conditions due to limited dataset diversity and computational constraints.  
- **Final Approach:** Adoption of a pre-trained FaceNet model paired with MTCNN for face detection, significantly improving accuracy and real-time performance for attendance tracking.

---

## Repository Contents

- `siamese-network.ipynb`: Notebook containing the full pipeline for training the Siamese Neural Network, including data preprocessing, model architecture, training, and evaluation.

---

## Key Features

- Custom dataset creation combining own captured images and LFW dataset
- Siamese network architecture using a pre-trained ResNet50 as the base feature extractor
- Enhanced training techniques: Batch Normalization, LeakyReLU, Dropout, ReduceLROnPlateau
- Implementation of L2 normalization and distance layers for similarity learning
- Performance evaluation and comparison on real and unseen face images

---

## Limitations & Lessons Learned

- The trained Siamese Neural Network struggled to generalize to diverse and unseen faces, especially under varying lighting and pose conditions.
- Computational resource constraints limited dataset size and model complexity.
- These challenges motivated switching to a more robust pre-trained FaceNet model, which yielded superior results for the attendance system.

---

## References

- Nicholas Renotte’s work on facial verification with Siamese Networks  
  [GitHub Notebook](https://github.com/nicknochnack/FaceRecognition/blob/main/Facial%20Verification%20with%20a%20Siamese%20Network%20-%20Final.ipynb)  
  [YouTube Video](https://youtu.be/bK_k7eebGgc)

---

## Dataset

- Custom dataset combined with Labeled Faces in the Wild (LFW)  
- Available on Kaggle: [Dataset Link](https://www.kaggle.com/datasets/abdullahnawaz470/dataset-faces/data)

---

## How to Run

1. Clone the repository  
2. Open the `siamese-network.ipynb` notebook in a Jupyter environment or Kaggle notebook  
3. Ensure GPU acceleration is enabled for faster training  
4. Follow the notebook cells sequentially to preprocess data, train the model, and evaluate results
