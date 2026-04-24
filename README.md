# Linear Regression from Scratch

## Overview
This project implements a simple linear regression model from scratch using Python and NumPy. The goal is to understand how machine learning models learn by manually building prediction, loss calculation, and gradient descent without using ML libraries.

---

## What This Project Does
- Predicts values using a linear model: y = wx + b
- Measures error using Mean Squared Error (MSE)
- Optimizes parameters using gradient descent
- Learns the best-fit line for a given dataset

---

## Key Concepts

**Weight (w)**  
Controls the slope of the line

**Bias (b)**  
Shifts the line up or down

**Loss (MSE)**  
Measures how far predictions are from actual values

**Gradient Descent**  
Iteratively updates parameters to reduce error

---

## Tech Stack
- Python
- NumPy
- Matplotlib (for visualization)

---

## Project Structure
├── main.py # Core implementation
├── data.py # Dataset (optional)
└── README.md


---

## How It Works
1. Initialize parameters (w, b)
2. Make predictions using current parameters
3. Compute loss (MSE)
4. Calculate gradients (dw, db)
5. Update parameters using gradient descent
6. Repeat until loss decreases

---

## Example Output
- Model converges to approximate values:
  - w ≈ 2  
  - b ≈ 3  
- The predicted line closely fits the data points

---

## How to Run
```bash
python main.py

Why This Project

This project focuses on building intuition behind machine learning by avoiding high-level libraries like Scikit-learn and implementing the core logic manually.
