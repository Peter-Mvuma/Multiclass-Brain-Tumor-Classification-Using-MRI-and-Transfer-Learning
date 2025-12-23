# Multiclass-Brain-Tumor-Classification-Using-MRI-and-Transfer-Learning

**Project Decription**
End-to-end deep learning pipeline for multiclass brain tumor classification using MRI. Implements ResNet-50 transfer learning with callback-guided optimization, achieving 84% accuracy and strong performance on the challenging meningioma class.

**Project Overview**
Accurate classification of brain tumor types from MRI scans is essential for clinical decision-making, treatment planning, and prognosis. Manual interpretation of MRI images is time-consuming and subject to inter-reader variability, particularly when tumor types exhibit subtle visual differences.

This project presents an end-to-end deep learning pipeline for multiclass brain tumor classification using contrast-enhanced T1-weighted MRI images. The workflow progresses from raw .mat / .h5 data extraction to model training, evaluation, and optimization using transfer learning with ResNet-50.

The final selected model — a callback-guided, fine-tuned ResNet-50 — achieves strong, balanced performance across all tumor classes, with substantial improvements in the historically challenging meningioma class.

**Dataset**

Source: Figshare Brain Tumor MRI Dataset

Total Images: 3,064 MRI slices

Tumor Classes:

Glioma: 1,426

Pituitary: 930

Meningioma: 708

Image Type: Contrast-enhanced T1-weighted MRI

Original Format: .mat / .h5

Converted Format: Grayscale PNG

Data Splits (Stratified)
Split	Images
Training	2,144
Validation	460
Test	460
**Data Preprocessing Pipeline**

The dataset required extensive reconstruction and standardization before model training.

Key preprocessing steps:

Conversion of .mat / .h5 files to PNG images

Min-max normalization (pixel range 0–255 → 0–1)

Uniform resizing to 224 × 224

Metadata creation and stratified splitting

On-the-fly data augmentation:

±20° rotation

Zoom (up to 0.15)

Width/height shifts (0.1)

Horizontal flips

This ensured consistent intensity scaling, reduced overfitting, and improved model generalization.

**Model Development Strategy**

The project followed a progressive, multi-phase modeling approach:

**Phase 1 — Baseline CNN**

Simple convolutional architecture trained from scratch

Served as a performance benchmark

**Phase 2 — ResNet-50 Transfer Learning**

ImageNet-pretrained ResNet-50 backbone

Two-stage training:

Frozen backbone

Selective unfreezing of top layers

**Phase 3 — Callback-Guided ResNet-50 (Final Model)**

Added training control via callbacks:

ModelCheckpoint

ReduceLROnPlateau

EarlyStopping

Achieved the best balance of accuracy, stability, and class-level performance

**Phase 4 — ResNet-50 + CBAM + Grad-CAM**

Incorporated attention (CBAM) and explainability (Grad-CAM)

Improved interpretability but did not outperform Phase 3 in accuracy

**Model Performance Comparison**

Model	Accuracy	Macro-F1	F1 (Glioma)	F1 (Meningioma)	F1 (Pituitary)

#Baseline CNN	0.78	0.71	0.87	0.42	0.90

#ResNet-50 TL	0.79	0.75	0.80	0.55	0.89

#Callback-Guided ResNet-50 	0.84	0.83	0.83	0.73	0.92

#ResNet-50 + CBAM	0.79	0.75	0.84	0.49	0.88

Additional metrics (Final Model):

Macro-AUC: 0.933

Micro-AUC: 0.923

**Final Model Selection**

**Model 3: Callback-Guided ResNet-50 was selected as the final model due to**

Highest overall accuracy and macro-F1

83% improvement in meningioma F1 (0.40 → 0.73)

Stable convergence without overfitting

Balanced performance across all tumor types

Strong readiness for clinical and research workflows

**Evaluation & Analysis**

Confusion matrices and classification reports for all phases

ROC curves (per-class, micro, macro)

Training/validation accuracy and loss curves

Grad-CAM visualizations (Phase 4) for interpretability analysis

**Tech Stack**

Python

TensorFlow / Keras

NumPy, Pandas

scikit-learn

Matplotlib / Seaborn

SciPy, h5py

Apache Spark (for large-scale preprocessing)

# Author: Peter Mvuma
Course: Introduction to Big Data Analytics (SAT 5165)
Institution: Michigan Technological University
Project Type: Academic / Research Project
# Acknowledgement
Data Source: Brain Tumor MRI Dataset (Figshare) — https://doi.org/10.6084/m9.figshare.1512427
