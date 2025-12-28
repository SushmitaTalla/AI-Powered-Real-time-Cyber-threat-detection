# AI-Powered-Real-time-Cyber-threat-detection
AI powered anomaly detection to detect real time cyber security threats using lightweight anomaly detection system with two phases.
## Project Summary

This project presents an innovative AI-powered real-time cyber threat detection system that addresses critical challenges in modern network security. By implementing a two-phase detection architecture combining Autoencoder-based anomaly detection with Random Forest multi-class classification, the system provides both high-level threat alerting and granular attack classification capabilities.

A key innovation lies in the application of Conditional Tabular Generative Adversarial Networks (CTGAN) to address the severe class imbalance inherent in the CICIDS2017 dataset. By generating synthetic attack samples, the system ensures balanced training data, resulting in an unbiased model capable of detecting minority attack classes with high accuracy while maintaining acceptable false positive rates.

The hierarchical detection approach leverages an Autoencoder in Phase 1 to identify anomalous patterns indicative of attacks versus benign traffic, followed by a Random Forest classifier in Phase 2 for precise attack type classification. This design combines the anomaly detection capabilities of autoencoders with the robust multi-class classification performance of ensemble methods, making the system suitable for real-time deployment in high-throughput network environments while providing the actionable intelligence security teams need for effective incident response.

---

## Table of Contents

- [Introduction/Background](#introductionbackground)
- [Data Background](#data-background)
- [Objectives](#objectives)
- [Methodology](#methodology)
- [Challenges](#challenges)
- [Results](#results)

---

## Introduction/Background

In today's interconnected digital landscape, cybersecurity threats are becoming increasingly sophisticated and prevalent, posing significant risks to organizations and individuals alike. Traditional signature-based intrusion detection systems struggle to keep pace with evolving attack patterns and zero-day exploits. The emergence of machine learning and artificial intelligence offers a promising avenue for developing adaptive, intelligent cyber threat detection systems capable of identifying both known and novel attack vectors in real-time.

This project addresses the critical need for an intelligent, automated system that can detect and classify cyber threats with high accuracy and minimal latency. By leveraging advanced machine learning techniques including generative models and ensemble methods, the system aims to provide robust protection against various types of network-based attacks while maintaining low false positive rates that are essential for operational efficiency.

---

## Data Background

The project utilizes the **CICIDS2017 dataset**, a comprehensive and widely-recognized benchmark dataset for intrusion detection research developed by the Canadian Institute for Cybersecurity. This dataset contains network traffic data captured over five days and includes both benign traffic and various contemporary attack scenarios.

### Dataset Characteristics

- Contains labeled network flows with **80+ features** extracted from raw packet data
- Includes multiple attack categories: **DoS/DDoS attacks, PortScan, Brute Force, Web attacks, Infiltration, Botnet traffic**, and others
- Represents realistic network traffic patterns from modern network environments
- Features include flow duration, packet statistics, protocol information, and behavioral characteristics

### Key Challenge

The dataset exhibits significant **class imbalance**, with benign traffic instances vastly outnumbering attack instances. This imbalance can lead to biased model training where the classifier becomes overly optimized for the majority class (benign traffic) while failing to adequately learn patterns associated with attack traffic, resulting in poor detection rates for actual threats.

---

## Objectives

The primary objectives of this project are:

1. **Develop a Two-Phase Detection Architecture:** Create an intelligent, hierarchical system that first identifies whether network traffic is malicious or benign, then classifies the specific type of attack if malicious activity is detected.

2. **Address Class Imbalance:** Implement data augmentation techniques using Conditional Tabular Generative Adversarial Networks (CTGAN) to generate synthetic attack samples, ensuring balanced representation across all classes for unbiased model training.

3. **Achieve High Detection Accuracy:** Maximize true positive rates for attack detection while minimizing false positives that could overwhelm security analysts or disrupt legitimate network operations.

4. **Enable Real-Time Detection:** Design a system architecture capable of processing network traffic in real-time with minimal latency, making it suitable for deployment in production environments.

5. **Provide Actionable Intelligence:** Not only detect attacks but also classify them into specific categories, enabling appropriate and targeted incident response measures.

---

## Methodology

The project employs a systematic approach combining data preprocessing, augmentation, and a novel two-phase detection architecture:

### Phase 1: Data Preparation and Augmentation

- **Exploratory Data Analysis:** Understanding feature distributions and class imbalance ratios
- **Feature Engineering and Selection:** Identifying the most discriminative attributes for threat detection
- **CTGAN Implementation:** Conditional Tabular Generative Adversarial Network to synthesize realistic attack samples
  - The CTGAN model learns the underlying distribution of minority attack classes
  - Generates synthetic instances that preserve statistical properties and inter-feature relationships
- **Dataset Balancing:** Achieving equal representation between benign and attack classes to reduce model bias

### Phase 2: Two-Phase Detection System

#### Phase 1 - Binary Classification (Attack vs. Benign)

- **Autoencoder-based architecture** deployed as the first-stage detector
- Trained on balanced dataset to learn the normal patterns of benign network traffic
- Detects anomalies based on reconstruction error: benign traffic is reconstructed accurately (low error), while attack traffic produces high reconstruction error
- Selected for its unsupervised learning capability and effectiveness in anomaly detection
- Acts as a gating mechanism to filter benign traffic and route suspicious traffic for further analysis

#### Phase 2 - Multi-Class Attack Classification

- **Random Forest classifier** for identifying specific attack types
- Ensemble learning approach that combines multiple decision trees to achieve robust classification
- Trained only on attack samples to distinguish between DoS, DDoS, PortScan, Brute Force, Web attacks, and other threat categories
- Selected for its high accuracy, resistance to overfitting, and ability to handle high-dimensional feature spaces
- Provides feature importance rankings to understand which network characteristics are most indicative of each attack type

### Model Training and Validation

- Dataset split into training, validation, and test sets with stratified sampling
- Hyperparameter optimization using cross-validation techniques
- Performance evaluation using metrics including **accuracy, precision, recall, F1-score, and confusion matrices**
- Special attention to false positive and false negative rates given their operational implications

---

## Challenges

Throughout the development of this system, several significant challenges were encountered and addressed:

### 1. Severe Class Imbalance

The most prominent challenge was the substantial disparity between benign and attack instances in the CICIDS2017 dataset. This imbalance risked creating a model that achieved high overall accuracy by simply predicting the majority class while failing to detect actual attacks. The CTGAN-based augmentation strategy was essential but required careful tuning to ensure synthetic samples were realistic and didn't introduce artifacts that could mislead the classifier.

### 2. Generating High-Quality Synthetic Data

Training the CTGAN to produce realistic attack samples that preserved the complex inter-feature relationships and statistical properties of real network traffic proved challenging. Poorly generated synthetic data could introduce noise and degrade model performance rather than improve it. Validation of synthetic data quality through statistical tests and domain expert review was necessary.

### 3. Feature Engineering and Dimensionality

With over 80 features in the CICIDS2017 dataset, identifying the most relevant features for threat detection while avoiding the curse of dimensionality required systematic feature selection and importance analysis. Irrelevant or redundant features could increase computational overhead and reduce model generalization.

### 4. Two-Phase Architecture Optimization

Designing an effective handoff between the autoencoder-based anomaly detection phase and the Random Forest classification phase required careful consideration. The system needed to set an appropriate reconstruction error threshold in Phase 1 to minimize false negatives (missing attacks) while ensuring Phase 2 could accurately classify the specific attack types without being overwhelmed by false positives from Phase 1.

### 5. Autoencoder Design for Anomaly Detection

Configuring the autoencoder architecture for optimal attack detection presented challenges in determining the appropriate encoding dimension, number of layers, activation functions, and reconstruction loss functions. The model needed to learn robust representations of benign traffic patterns while remaining sensitive enough to detect various types of attacks as anomalies based on reconstruction error.

### 6. Random Forest Optimization for Attack Classification

Tuning the Random Forest classifier required careful selection of hyperparameters including the number of trees, maximum depth, minimum samples per leaf, and feature subset size. The model needed to achieve high multi-class classification accuracy across diverse attack types while maintaining computational efficiency for real-time operation.

### 7. Real-Time Performance Requirements

Balancing model complexity with inference speed was critical for real-time deployment. The autoencoder's forward pass for anomaly detection and the Random Forest's ensemble prediction both needed to operate within acceptable latency bounds to process high-volume network traffic without introducing bottlenecks.

### 8. Evaluation and Validation

Establishing appropriate evaluation metrics and validation strategies that accurately reflected real-world operational conditions was challenging. Standard accuracy metrics could be misleading in the context of imbalanced threat detection, requiring focus on precision-recall tradeoffs and cost-sensitive evaluation frameworks.

---

### Requirements

- Python 3.8+
- TensorFlow/Keras
- Scikit-learn
- CTGAN
- Pandas
- NumPy
- Matplotlib/Seaborn

---

## Results

- **Phase 1 Accuracy:** 99%
- **Phase 2 Accuracy:** 99%
- **Precision:** 98%
- **Recall:** 98%
- **F1-Score:** 98%

---

## Acknowledgments

- Canadian Institute for Cybersecurity for the CICIDS2017 dataset
- CTGAN developers for the data augmentation framework
- The open-source community for various tools and libraries

---

## Contact
Talla Sushmita- talla.sushmita@gmail.com

Project Link: [https://github.com/SushmitaTalla/AI-Powered-Real-time-Cyber-threat-detection](https://github.com/SushmitaTalla/AI-Powered-Real-time-Cyber-threat-detection)


