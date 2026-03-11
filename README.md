# EEG-Based Schizophrenia Detection using EMD, LMS, Machine Learning and Deep Learning

This repository presents an EEG-based schizophrenia detection project combining **signal processing**, **feature engineering**, **traditional machine learning**, and **deep learning**.

The core idea is to improve EEG classification by comparing three preprocessing strategies:

- **Approach 1:** Empirical Mode Decomposition (**EMD**) only
- **Approach 2:** Least Mean Squares (**LMS**) adaptive filtering only
- **Approach 3:** Combined **EMD + LMS**

After preprocessing, the project extracts both:

- **amplitude/statistical features**  
  (energy, standard deviation, Shannon entropy, differential entropy)
- **connectivity features**  
  (PLV, PLI)

These features are then evaluated using:

- **SVM**
- **Random Forest**
- **1D-CNN**
- **LSTM**


---

## Project Motivation

Schizophrenia diagnosis remains difficult, and EEG is an attractive modality because it is **non-invasive**, relatively accessible, and able to capture brain activity patterns linked to psychiatric disorders. However, EEG signals are highly **non-linear**, **non-stationary**, and often corrupted by artifacts, which makes direct classification challenging. The project therefore investigates whether suitable preprocessing and feature extraction strategies can improve schizophrenia detection from EEG recordings.

---

## Highlights

- EEG-based schizophrenia classification on **two datasets**
- Comparison of **EMD**, **LMS**, and **EMD + LMS**
- Extraction of **statistical**, **entropy-based**, and **phase-locking** features
- Evaluation with **SVM**, **Random Forest**, **1D-CNN**, and **LSTM**
- Strong results with **Random Forest**
- Promising deep learning results with **1D-CNN** and **LSTM** on Dataset 2 :contentReference[oaicite:4]{index=4}

---

## Methodology Overview

The general pipeline of the project is:

**Raw EEG signals**  
→ **Preprocessing / signal enhancement**  
→ **Feature extraction**  
→ **Classification**  
→ **Evaluation**


---

## Preprocessing Approaches

### Approach 1 — EMD only
This approach uses **Empirical Mode Decomposition** to decompose EEG signals into **Intrinsic Mode Functions (IMFs)**.  
It is particularly suited for EEG because EEG is inherently non-linear and non-stationary. The idea is to isolate oscillatory modes and extract more discriminative features from IMFs than from raw EEG segments.

### Approach 2 — LMS only
This approach applies an **LMS adaptive filter** directly to EEG channels, typically using **FP1** as a reference channel.  
The goal is to reduce artifacts, especially frontal noise and ocular activity, before extracting features. This choice may remove useful frontal information if FP1 contains task-relevant EEG activity.

### Approach 3 — EMD + LMS
This hybrid approach first decomposes each EEG segment into IMFs using **EMD**, then applies **LMS filtering** to the IMFs.  
The rationale is to combine:

- EMD for handling non-stationarity and isolating oscillatory modes
- LMS for adaptive artifact reduction

The report concludes that this combination often provides the best denoising / feature extraction trade-off, especially with Random Forest on Dataset 1.

---

## Features

### 1) Statistical / amplitude-based features
The project uses several classical descriptors:

- **Energy**
- **Standard deviation**
- **Shannon entropy**
- **Differential entropy**

These are used to capture amplitude, variability, and complexity of EEG segments or IMFs. The differential entropy was used as an input representation for the 1D-CNN on Dataset 2. 

### 2) Connectivity / phase-locking features
The project also extracts phase-based connectivity descriptors:

- **PLV (Phase Locking Value)**
- **PLI (Phase Lag Index)**

These features quantify synchronization or preferred phase lag between EEG channels and are especially useful for investigating altered brain connectivity in schizophrenia. There are visible differences in the **alpha band** between healthy and schizophrenia subjects. 

---

## Datasets

### Dataset 1
Dataset 1 comes from the RepOD repository and includes:

- **14 schizophrenia patients**
- **14 healthy controls**
- **19 EEG channels**
- **250 Hz sampling frequency**
- **15 minutes recording per subject**
- **resting state, eyes closed** 

### Dataset 2
Dataset 2 is the **ASZED / NSzED** dataset from Zenodo and includes:

- **76 schizophrenia patients**
- **77 healthy controls**
- recordings from **two devices**
- **22 or 24 channels**
- **200 Hz or 256 Hz**
- multiple paradigms:
  - Rest
  - Arithmetic task
  - Auditory oddball
  - Fixed-frequency auditory stimulation / ASSR depending on scheme 

Because Dataset 2 mixes different acquisition protocols, preprocessing includes:

- channel unification to **24 channels**
- resampling to **256 Hz**
- segmentation into **10-second windows** with **50% overlap** 

---

## Models

### Traditional Machine Learning
- **Support Vector Machine (SVM)**
- **Random Forest**

### Deep Learning
- **1D-CNN**
- **LSTM**

The 1D-CNN is described as a model adapted to 1D time-series such as EEG, and the LSTM as a model designed to capture long-term temporal dependencies in sequential signals.

---

## Main Results

### Dataset 1
The project compares the three preprocessing approaches on Dataset 1 using two feature sets:

- **F1:** energy + standard deviation + Shannon entropy
- **F2:** PLV + PLI

#### Reported accuracies on Dataset 1

| Classifier | Approach 1 (EMD) F1 | Approach 1 (EMD) F2 | Approach 2 (LMS) F1 | Approach 2 (LMS) F2 | Approach 3 (EMD+LMS) F1 | Approach 3 (EMD+LMS) F2 |
|-----------|---------------------:|---------------------:|---------------------:|---------------------:|-------------------------:|-------------------------:|
| SVM | 74% | 81% | 68% | 67% | 70% | 63% |
| Random Forest | 93% | 85% | 89% | 74% | **94%** | 82% |

The best reported result on Dataset 1 is therefore:

- **94% accuracy** with **Random Forest**
- using **EMD + LMS**
- with **F1 statistical/amplitude features** 

### Dataset 2
For Dataset 2, the report indicates that Random Forest was prioritized because it was consistently stronger than SVM in earlier experiments.

#### Reported accuracies on Dataset 2

| Classifier | Approach 1 (EMD) F1 | Approach 1 (EMD) F2 | Approach 2 (LMS) F1 | Approach 2 (LMS) F2 | Approach 3 (EMD+LMS) F1 | Approach 3 (EMD+LMS) F2 |
|-----------|---------------------:|---------------------:|---------------------:|---------------------:|-------------------------:|-------------------------:|
| Random Forest | **95%** | 80% | 90% | 86% | 92% | 70% |

The best reported result on Dataset 2 is:

- **95% accuracy** with **Random Forest**
- using **EMD only**
- with **F1 statistical/amplitude features** 

### Deep Learning on Dataset 2
The report also explores two deep learning approaches:

#### 1D-CNN
The 1D-CNN is trained on **differential entropy features extracted from IMFs**.  
The training and validation curves remain controlled and that the network learns discriminative EEG patterns with good generalization.

#### LSTM
The LSTM is trained on **EMD-based IMF time-series representations**.  
It achieved an overall **F1-score around 0.88–0.89**, although some misclassifications remained. Besides, deeper versions of the LSTM did not improve performance and that larger datasets may be required.

## Python Dependencies

This project requires the following Python libraries:

numpy

pandas

scipy

matplotlib

seaborn

scikit-learn

mne

joblib

emd

EMD-signal

tensorflow

## Installation

Clone the repository:

git clone https://github.com/mallouliemna09-lang/eeg-schizophrenia-detection.git
cd eeg-schizophrenia-detection

Create a Python environment (recommended):

conda create -n eeg-schizo python=3.10

conda activate eeg-schizo

pip install -r requirements.txt

