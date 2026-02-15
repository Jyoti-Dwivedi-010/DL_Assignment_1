# DL_Assignment_1
# Stress-Testing of Convolutional Neural Networks on CIFAR-10

## Overview
This project studies the behaviour and reliability of a deep learning image classifier instead of evaluating it only using accuracy.  
A *ResNet-18* model is trained from scratch on the *CIFAR-10 dataset*, and the model is analysed using training curves, failure case detection, Grad-CAM explainability, and controlled improvements.

The aim of this assignment is to understand *how the model makes decisions* and whether those decisions are trustworthy.

---

## Objectives
- Train a CNN model on CIFAR-10
- Identify high-confidence misclassifications
- Visualize model reasoning using Grad-CAM & Saliency maps
- Improve robustness using augmentation and regularization
- Compare behaviour before and after improvements

---

## Dataset
*CIFAR-10*
- 60,000 RGB images (32×32 resolution)
- 10 object classes:
  - airplane
  - automobile
  - bird
  - cat
  - deer
  - dog
  - frog
  - horse
  - ship
  - truck

### Why CIFAR-10?
- Contains cluttered backgrounds
- Several visually similar classes (cat/dog, deer/horse)
- Helps reveal model bias and shortcut learning

---

## Model
*ResNet-18 (trained from scratch)*

### Modifications
- Adjusted first convolution layer for small images
- Dropout added in fully connected layer
- Label smoothing applied
- Learning rate scheduling used

### Why ResNet-18?
- Stable training due to residual connections
- Moderate complexity → easy analysis
- Good balance between performance and interpretability

---

## Training Configuration

| Parameter | Value |
|--------|------|
| Optimizer | Adam / AdamW |
| Scheduler | CosineAnnealing / OneCycleLR |
| Epochs | 40–50 |
| Batch Size | 128 |
| Label Smoothing | 0.05–0.1 |
| Dropout | 0.2–0.3 |

### Data Augmentation
- Random Crop
- Horizontal Flip
- Rotation
- Color Jitter
- Random Erasing

---

## Results

### Baseline Model
- High training accuracy
- Lower validation accuracy
- Overfitting observed

### Final Model
- *~91% Training Accuracy*
- *~90% Validation Accuracy*
- Improved generalization
- More stable predictions

---

## Failure Case Analysis
We examined predictions where the model was highly confident but wrong.

Examples:
- Airplane → Ship
- Cat → Bird
- Deer → Frog

### Observation
The model relied on *background textures and colours* instead of object structure.

---

## Explainability (Grad-CAM & Saliency Maps)

We visualized attention regions of the network.

*Findings*
- Model focused on background regions
- Ignored important object features
- Sometimes correct prediction but wrong reasoning

After augmentation:
- Attention shifted to object body
- Predictions became more reliable

---

## Key Learnings
- Accuracy alone cannot measure reliability
- CNNs learn shortcuts from dataset bias
- Explainability helps detect incorrect reasoning
- Data augmentation improves generalization and reasoning
- Behaviour analysis is important before deployment

---

## Conclusion
This assignment demonstrates that a deep learning model can achieve high accuracy while still making decisions for incorrect reasons.  
By analysing training behaviour, failure cases and visual explanations, we improved both performance and reliability of the model.  
Understanding model behaviour is essential for building trustworthy AI systems.

---

## Authors
- Jyoti Dwivedi (M25CSA010)
- Jahanvi Gajera (M25CSA012)
- Mamta Chauhan (M25CSA018)
- Akanksha Kapil (M25CSA033)

---

## How to Run
```bash
git clone <https://github.com/Jyoti-Dwivedi-010/DL_Assignment_1>
python train.py
