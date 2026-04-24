Linear Regression from Scratch
Overview

This project implements a simple linear regression model from scratch using Python and NumPy. The goal is to understand how machine learning models learn by manually building prediction, loss calculation, and gradient descent without using ML libraries.

What This Project Does
Predicts values using a linear model:
y=wx+b
Measures error using Mean Squared Error (MSE)
Optimizes parameters using gradient descent
Learns the best-fit line for a given dataset
Key Concepts
Weight (w): Controls the slope of the line
Bias (b): Shifts the line up or down
Loss (MSE): Measures how far predictions are from actual values
Gradient Descent: Iteratively updates parameters to reduce error
Tech Stack
Python
NumPy
Matplotlib (for visualization)
Project Structure
.
├── main.py          # Core implementation
├── data.py          # Dataset (optional)
└── README.md
How It Works
Initialize parameters (w, b)
Make predictions using current parameters
Compute loss (MSE)
Calculate gradients (dw, db)
Update parameters using gradient descent
Repeat until loss decreases
Example Output
Model converges to approximate values of:
w ≈ 2
b ≈ 3
The predicted line closely fits the data points
How to Run
python main.py
Why This Project

This project focuses on building intuition behind machine learning by avoiding high-level libraries like Scikit-learn and implementing the core logic manually.
