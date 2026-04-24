import numpy as np
import matplotlib.pyplot as plt

# y = 2x + 3 + noise
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = 2 * X + 3 + np.random.randn(len(X))

w = 0.0
b = 0.0
learning_rate = 0.01

def predict(X, w, b):
    return w * X + b

def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_gradients(X, y, y_pred):
    n = len(X)
    
    dw = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    return dw, db

for epoch in range(1000):
    y_pred = predict(X, w, b)
    
    loss = compute_loss(y, y_pred)
    
    dw, db = compute_gradients(X, y, y_pred)
    
    w -= learning_rate * dw
    b -= learning_rate * db
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("Final w:", w)
print("Final b:", b)



plt.scatter(X, y, label="Data")
plt.plot(X, predict(X, w, b), color='red', label="Model")
plt.legend()
plt.show()