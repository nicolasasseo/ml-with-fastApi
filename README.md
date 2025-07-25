# MNIST Digit Recognition with FastAPI

## Overview

This project is a simple web application for recognizing handwritten digits using the MNIST dataset and a machine learning model. The backend is built with FastAPI and serves predictions via a REST API, while the frontend is a minimal HTML page for uploading images and displaying results.

## Motivation

Handwritten digit recognition is a classic problem in machine learning and computer vision. This project demonstrates how to:

- Train a machine learning model (Random Forest) on the MNIST dataset
- Serve the model using a FastAPI backend
- Build a simple web interface for user interaction

It is intended as a learning resource for combining ML with modern Python web frameworks.

## How to Use

### 1. Install Requirements

Make sure you have Python 3.7+ installed. Install dependencies:

```bash
pip install fastapi uvicorn scikit-learn pillow numpy
```

### 2. Train the Model

Run the following command to train the model and save it as `mnist_model.pkl`:

```bash
python train_model.py
```

### 3. Start the FastAPI Server

Run the API server with:

```bash
uvicorn main:app --reload
```

The server will be available at `http://127.0.0.1:8000`.

### 4. Use the Web Interface

Open `index.html` in your browser. Upload an image of a handwritten digit (preferably 28x28 pixels, grayscale, or similar to MNIST style) and click Upload. The prediction will be displayed on the page.

## File Structure

- `train_model.py`: Trains and saves the Random Forest model
- `main.py`: FastAPI backend serving predictions
- `index.html`: Simple frontend for image upload and result display
- `mnist_model.pkl`: Saved model file (generated after training)

## Notes

- The backend expects images similar to MNIST (28x28, grayscale, white digit on black background). Other formats may reduce accuracy.
- CORS is enabled for all origins for easy local development.
