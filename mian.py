import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# =========================
# 1. CHARGEMENT DES DONNÉES
# =========================
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Convertir en numpy (IMPORTANT)
X = X.values
y = y.values.astype(int)

# Normalisation (0 → 1)
X = X / 255.0

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# =========================
# 2. AFFICHER UNE IMAGE
# =========================
plt.imshow(X[0].reshape(28,28), cmap='gray')
plt.title(f"Label: {y[0]}")
plt.show()

# =========================
# 3. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000)

# =========================
# 4. ONE HOT
# =========================
def one_hot(y):
    Y = np.zeros((len(y), 10))
    Y[np.arange(len(y)), y] = 1
    return Y

Y_train = one_hot(y_train)

# =========================
# 5. INITIALISATION
# =========================
A = np.random.randn(10, 784) * 0.01
b = np.zeros((10, 1))

# =========================
# 6. SOFTMAX
# =========================
def softmax(o):
    exp_o = np.exp(o - np.max(o))
    return exp_o / np.sum(exp_o)

# =========================
# 7. ENTRAÎNEMENT
# =========================
lr = 0.1

for i in range(1000):
    idx = np.random.randint(0, len(X_train))
    
    x = X_train[idx].reshape(784,1)
    y_true = Y_train[idx].reshape(10,1)
    
    # Forward
    o = A @ x + b
    P = softmax(o)
    
    # Gradient
    dA = (P - y_true) @ x.T
    db = (P - y_true)
    
    # Update
    A -= lr * dA
    b -= lr * db
    
    if i % 100 == 0:
        loss = -np.sum(y_true * np.log(P + 1e-9))
        print("Iteration", i, "loss:", loss)

# =========================
# 8. TEST
# =========================
correct = 0

for i in range(len(X_test)):
    x = X_test[i].reshape(784,1)
    o = A @ x + b
    pred = np.argmax(o)
    
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)
print("Accuracy:", accuracy)