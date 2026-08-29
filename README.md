# 🚀 DecodeLabs Internship — Data Science Portfolio

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Data%20Science-Projects-purple" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Internship-DecodeLabs-success" />
  <img src="https://img.shields.io/badge/Batch-2026-informational" />

</p>

<p align="center">
  <b>Practical Data Science & Machine Learning Projects</b>
</p>

---

## 📌 About This Repository

This repository contains my projects completed as part of the **DecodeLabs Data Science Industrial Training Program — Batch 2026**.

The projects cover different stages of the Data Science and Machine Learning workflow, from **data cleaning and feature engineering** to **supervised learning, unsupervised learning, and Natural Language Processing**.

The goal of these projects was to develop practical experience in transforming raw data into meaningful insights, building machine learning pipelines, evaluating models, and translating analytical results into useful conclusions.

---

## 📚 Table of Contents

- [📌 About This Repository](#-about-this-repository)
- [🗂️ Projects Overview](#️-projects-overview)
- [📊 Project 1 — Advanced EDA & Feature Engineering](#-project-1--advanced-eda--feature-engineering)
- [🛡️ Project 2 — Fraud Detection Pipeline](#️-project-2--fraud-detection-pipeline)
- [👥 Project 3 — Customer Segmentation](#-project-3--customer-segmentation)
- [💬 Project 4 — NLP & Sentiment Analysis](#-project-4--nlp--sentiment-analysis)
- [🛠️ Technologies & Libraries](#️-technologies--libraries)
- [🧠 Skills Demonstrated](#-skills-demonstrated)
- [📁 Repository Structure](#-repository-structure)
- [🎯 Learning Objective](#-learning-objective)

---

# 🗂️ Projects Overview

| # | Project | Area | Main Techniques |
|---|---|---|---|
| 01 | 📊 Advanced EDA & Feature Engineering | Data Preprocessing | Imputation, IQR, Encoding, Feature Engineering |
| 02 | 🛡️ Fraud Detection Pipeline | Supervised Learning | SMOTE, Logistic Regression, Random Forest |
| 03 | 👥 Customer Segmentation | Unsupervised Learning | PCA, K-Means, Elbow Method, Silhouette Score |
| 04 | 💬 NLP Sentiment Analysis | Natural Language Processing | TF-IDF, NLP Preprocessing, Naive Bayes, SVM |

---

# 📊 Project 1 — Advanced EDA & Feature Engineering

### 🧹 Turning messy data into a machine-learning-ready dataset

**Dataset:** Synthetic `ShopSphere` e-commerce customer dataset

**[📓 View Project 1 Notebook →](./Project1_Advanced_EDA_Feature_Engineering.ipynb)**

### 🎯 Objective

Transform a raw and intentionally messy dataset into a clean, structured dataset suitable for machine learning.

### 🔍 Key Work

- Missing-data analysis and decision-making
- Mean / Median / KNN imputation
- Group-wise conditional imputation
- IQR-based outlier detection
- Winsorization / outlier clipping
- One-hot encoding
- Multicollinearity detection
- Correlation analysis
- Feature engineering

### ⚙️ Engineered Features

The project creates four additional predictive features:

- `recency_frequency_score`
- `income_to_spend_ratio`
- `loyalty_value_index`
- `is_high_value_customer`

### 💡 Key Takeaway

The project demonstrates how preprocessing decisions should be driven by the **type and extent of the data problem**, rather than applying one technique blindly to every column.

---

# 🛡️ Project 2 — Fraud Detection Pipeline

### 🚨 Detecting fraudulent transactions in highly imbalanced data

**Dataset:** Synthetic `CreditGuard` transaction dataset

**[📓 View Project 2 Notebook →](./Project2_Fraud_Detection_Pipeline.ipynb)**

### 🎯 Objective

Build a classification pipeline capable of detecting fraudulent transactions while properly handling extreme class imbalance and preventing data leakage.

### 🔍 Key Work

- Highly imbalanced classification
- Stratified train/test splitting
- SMOTE oversampling
- Leak-free preprocessing pipelines
- Logistic Regression
- Random Forest
- GridSearchCV
- Stratified K-Fold Cross-Validation
- ROC-AUC optimization
- Precision / Recall / F1 evaluation
- Decision-threshold tuning
- Data leakage demonstration

### 🧠 Important Design Decision

Instead of relying on accuracy, the project focuses on:

> **Precision + Recall + F1 + ROC-AUC**

This is particularly important for fraud detection because a model can achieve high accuracy simply by predicting almost every transaction as legitimate.

### 🔐 Leakage Prevention

SMOTE and scaling are applied **inside the training pipeline**, ensuring that synthetic samples and preprocessing do not leak information from the test set into model training.

### 💡 Key Takeaway

The project focuses not only on building a classifier, but on designing a **proper, leak-free machine learning workflow** for an imbalanced classification problem.

---

# 👥 Project 3 — Customer Segmentation

### 🛍️ Discovering hidden customer groups using unsupervised learning

**Dataset:** Synthetic `RetailPulse` customer dataset

**[📓 View Project 3 Notebook →](./Project3_Customer_Segmentation.ipynb)**

### 🎯 Objective

Discover hidden customer groups from behavioral and monetary data without providing the model with predefined customer labels.

### 🔍 Key Work

- Feature scaling
- Dimensionality reduction
- Principal Component Analysis (PCA)
- Explained variance analysis
- K-Means clustering
- Elbow Method
- Kneedle algorithm
- Silhouette Score
- Silhouette analysis
- 2D / 3D cluster visualization
- Cluster centroid reconstruction
- Business persona interpretation

### 🔄 Methodology

The project follows the pipeline:

```text
Scale
   ↓
Compress
   ↓
Cluster
   ↓
Translate
```

PCA is used to reduce the dimensionality of the 24 numerical behavioral and monetary features, while K-Means is used to identify customer groups.

### 👤 Discovered Customer Personas

The analysis translates the discovered clusters into four business-oriented personas:

- 🏆 **High-Value Trendsetters**
- 💎 **Affluent Conservatives**
- 🛍️ **Budget-Conscious Explorers**
- 🧊 **Conservative Minimizers**

### 💡 Key Takeaway

The project demonstrates how unsupervised machine learning can move beyond mathematical clusters and translate them into **interpretable business segments**.

---

# 💬 Project 4 — NLP & Sentiment Analysis

### 📝 Teaching a machine to understand product-review sentiment

**Dataset:** Synthetic product-review dataset

**[📓 View Project 4 Notebook →](./Project4_NLP_Sentiment_Analysis.ipynb)**

### 🎯 Objective

Build an NLP pipeline that classifies product reviews as **Positive** or **Negative**.

### 📊 Dataset

The project uses:

- 600 product reviews
- 300 Positive reviews
- 300 Negative reviews
- 18 product categories

The reviews intentionally contain realistic noise such as HTML tags, punctuation, capitalized words, and negation phrases.

### 🔍 NLP Pipeline

```text
Raw Text
   ↓
Text Cleaning
   ↓
Tokenization
   ↓
Negation-Safe Stop-Word Removal
   ↓
POS Tagging
   ↓
Lemmatization
   ↓
TF-IDF Vectorization
   ↓
Machine Learning Model
   ↓
Sentiment Prediction
```

### 🧠 Key Techniques

- Text normalization
- Tokenization
- Negation-safe stop-word removal
- POS tagging
- POS-guided lemmatization
- TF-IDF
- Unigrams + Bigrams
- SciPy CSR sparse matrices
- Multinomial Naive Bayes
- Complement Naive Bayes
- Linear SVM
- Confusion Matrix
- Classification Report

### ⚠️ Special NLP Challenge

The project specifically addresses the **negation problem**.

For example:

```text
"I am not happy."
```

Removing the word `not` can completely reverse the meaning of the sentence.

The preprocessing pipeline therefore preserves important negation words rather than blindly removing them.

### 💡 Key Takeaway

The project demonstrates an end-to-end NLP workflow while also examining limitations such as **sarcasm**, which can be difficult for traditional TF-IDF-based models to understand.

---

# 🛠️ Technologies & Libraries

### 💻 Programming

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)

### 📊 Data Analysis & Visualization

- Pandas
- NumPy
- Matplotlib
- Seaborn

### 🤖 Machine Learning

- Scikit-learn
- imbalanced-learn
- K-Means
- PCA
- Logistic Regression
- Random Forest
- Naive Bayes
- Linear SVM

### 📝 Natural Language Processing

- NLTK
- WordNet
- TF-IDF

### 📓 Development Environment

- Jupyter Notebook

---

# 🧠 Skills Demonstrated

Through these projects, I practiced:

### 📊 Data Analysis

- Exploratory Data Analysis
- Data Cleaning
- Missing Data Handling
- Outlier Detection
- Statistical Imputation
- Correlation Analysis

### 🧮 Feature Engineering

- Feature creation
- One-hot encoding
- Multicollinearity detection
- Feature selection

### 🤖 Machine Learning

- Supervised Learning
- Unsupervised Learning
- Classification
- Clustering
- Dimensionality Reduction
- Hyperparameter Tuning
- Cross-Validation

### ⚖️ Model Evaluation

- Precision
- Recall
- F1 Score
- ROC-AUC
- Silhouette Score
- Confusion Matrix

### 📝 NLP

- Text preprocessing
- Tokenization
- Lemmatization
- TF-IDF
- N-grams
- Sentiment classification

### 🔐 Machine Learning Best Practices

- Data leakage prevention
- Stratified splitting
- Pipeline-based preprocessing
- Handling imbalanced datasets
- Threshold tuning
- Honest model limitation analysis

---

# 📁 Repository Structure

```text
DecodeLabs-Internship/
│
├── 📓 Project1_Advanced_EDA_Feature_Engineering.ipynb
├── 📓 Project2_Fraud_Detection_Pipeline.ipynb
├── 📓 Project3_Customer_Segmentation.ipynb
├── 📓 Project4_NLP_Sentiment_Analysis.ipynb
│
└── 📄 README.md
```

---

# 🎯 Learning Objective

These projects were designed to provide practical experience across multiple areas of the Data Science lifecycle:

```text
                    DATA SCIENCE WORKFLOW

                          Raw Data
                             │
                             ▼
                    🧹 Data Cleaning
                             │
                             ▼
                 🔍 Exploratory Analysis
                             │
                             ▼
                 ⚙️ Feature Engineering
                             │
                             ▼
                  🤖 Machine Learning
                       /           \
                      /             \
             Supervised          Unsupervised
                 │                    │
                 ▼                    ▼
             Prediction           Segmentation
                      \             /
                       \           /
                        ▼         ▼
                       📊 Evaluation
                             │
                             ▼
                    💡 Business Insight
```

The overall objective was to develop the ability to approach data problems systematically, select appropriate techniques, evaluate results critically, and communicate findings clearly.

---

## ⭐ Project Highlights

| Project | Highlight |
|---|---|
| 📊 EDA & Feature Engineering | Multiple imputation strategies + outlier handling + feature creation |
| 🛡️ Fraud Detection | SMOTE + leak-free pipelines + threshold tuning |
| 👥 Customer Segmentation | PCA + K-Means + mathematically justified clustering |
| 💬 NLP Sentiment Analysis | Negation-safe preprocessing + TF-IDF + multiple classifiers |

---

<p align="center">

### 🚀 From Raw Data → Machine Learning → Business Insight

</p>

<p align="center">
  <i>DecodeLabs Industrial Training — Batch 2026</i>
</p>
