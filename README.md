# House-price-prediction-model-using-Bayut-Dataset

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This repository contains a high-performance machine learning pipeline designed to predict property prices in the Egyptian real estate market. Using a dataset of ~50,000 listings scraped from **Bayut**, the model focuses on capturing hyper-local pricing trends in Cairo and Giza.

The project demonstrates an end-to-end DS workflow: from raw data cleaning and feature engineering to handling power-law distributions and deploying a gradient-boosted regressor.

## 📊 Performance Summary
The model demonstrates exceptional generalization, with a minimal delta between validation and test performance:

* **Training $R^2$ Score**: 89%
* **Validation $R^2$ Score**: 83%
* **Test $R^2$ Score**: 82% (Verified on unseen data)
* **Mean Absolute Error (MAE)**: ~1.8M EGP

## 🛠️ Technical Methodology

### 1. Feature Engineering (The "Hyper-Local" Approach)
* **Location Deconstruction**: Raw location strings were split into specific `regions` and broader `cities`. This allowed the model to learn the premium associated with specific compounds (e.g., *Mountain View iCity*, *Sodic*) versus general city averages.
* **Down Payment Logic**: Engineered a `dp_ratio` feature to understand the relationship between financial flexibility and total listing price.

### 2. Data Optimization
* **Log Transformation**: Real estate prices are heavily skewed. We applied a logarithmic transformation $y' = \ln(1 + y)$ to stabilize variance and achieve a near-normal distribution for the loss function.
* **Outlier Clipping**: Listings below the 1st and above the 99th percentile were removed to protect the model from data entry errors and extreme "one-off" luxury outliers.

### 3. Model Architecture
We utilized the **XGBoost (eXtreme Gradient Boosting)** regressor with a Histogram-based tree method (`hist`) for memory efficiency.
* **Regularization**: Applied `reg_alpha` and `min_child_weight` to enforce conservative splitting and prevent overfitting.
* **Optimization**: Implemented **Early Stopping** to halt training at the point of optimal validation loss.

## 🚀 Installation & Usage

### Requirements
* Python 3.13+
* Pandas, NumPy
* XGBoost
* Scikit-Learn
* Matplotlib/Seaborn

### Execution
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/egypt-real-estate-prediction.git](https://github.com/YourUsername/egypt-real-estate-prediction.git)
