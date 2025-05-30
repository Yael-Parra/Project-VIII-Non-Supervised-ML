# 🍄 Unsupervised Machine Learning Workshop with the Mushroom Dataset

This repository contains a practical workshop focused on **unsupervised machine learning**, utilizing **PCA** (Principal Component Analysis) and **Clustering (K-Means)** techniques. It also includes a comparative analysis with a supervised model (Random Forest).

We will be working with the **Mushroom Dataset**, a well-known educational dataset containing information about various types of mushrooms, including their classification as edible or poisonous.

## 🧠 Workshop Objectives

* Load and explore a complex categorical dataset.
* Handle null values and remove uninformative columns.
* Encode categorical variables using **One-Hot Encoding**.
* Perform dimensionality reduction with **PCA** (Principal Component Analysis).
* Apply **K-Means Clustering** to detect hidden structures within the data.
* Compare the performance of the unsupervised model with a supervised model (Random Forest).

## 🚀 Getting Started

To run the code in this repository, you'll need to access the Mushroom Dataset. You can find and download it from the UCI Machine Learning Repository:

[https://archive.ics.uci.edu/dataset/73/mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)

Make sure to place the dataset in the appropriate directory as expected by the `V1.ipynb` notebook, or modify the code to point to your dataset's location.

## 📄 Code Structure

The core of this workshop is presented in the `V1.ipynb` Jupyter Notebook. It sequentially covers:

1.  **Data Loading and Initial Exploration:** Steps to load the dataset and get a first look at its structure and content.
2.  **Data Preprocessing:** Handling missing values, dropping irrelevant columns, and performing One-Hot Encoding for categorical features.
3.  **Dimensionality Reduction with PCA:** Applying PCA to reduce the number of features while retaining important information.
4.  **K-Means Clustering:** Implementing K-Means to group similar mushroom instances based on their characteristics.
5.  **Supervised Model (Random Forest) for Comparison:** Training and evaluating a Random Forest classifier to serve as a benchmark for the unsupervised approaches.
6.  **Results and Analysis:** Comparing the outcomes of the unsupervised and supervised models.

## ⚙️ Requirements

To run the notebook, you'll need to have the following Python libraries installed:

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `scipy`
* `kaggle` (for Kaggle API if you choose to download data via API, though direct download is also an option)

You can install them using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy kaggle
