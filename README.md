# 🌸 Iris Flower Species Predictor

## Overview
This project is a Machine Learning web application that predicts the species of an Iris flower (Setosa, Versicolor, or Virginica) based on its physical measurements. It uses a **Support Vector Classifier (SVC)** built with Scikit-Learn to accurately classify the flowers based on sepal length, sepal width, petal length, and petal width.

The model is served through a lightweight **Flask** web API and is fully containerized using **Docker** for seamless deployment on platforms like Hugging Face Spaces.

## 🚀 Features
* **Machine Learning Model:** Utilizes a Support Vector Machine (SVC) for robust classification.
* **REST API:** A Flask-based backend that accepts physical measurements and returns the predicted flower species.
* **Production Ready:** Fully containerized with a `Dockerfile` and served via `gunicorn`.

## 🛠️ Tech Stack
* **Language:** Python 
* **Web Framework:** Flask
* **Machine Learning:** Scikit-Learn, Pandas
* **Deployment:** Docker, Gunicorn, Hugging Face Spaces

## 📁 Repository Structure
```text
├── app.py                 # The Flask web application / API endpoint
├── train_model.py         # Script to clean data, train the SVC model, and evaluate accuracy
├── model.joblib           # The saved Support Vector Classifier model
├── Dockerfile             # Container configuration for deployment
└── requirements.txt       # Python dependencies
