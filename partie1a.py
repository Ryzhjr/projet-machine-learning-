import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# =========================
# 1. CHARGEMENT DES DONNÉES
# =========================
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X.values / 255.0
y = y.values.astype(int)

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# =========================
# 2. AFFICHER UNE IMAGE
# =========================
plt.imshow(X[0].reshape(28, 28), cmap='gray')
plt.title(f"Label: {y[0]}")
plt.show()

# =========================
# 3. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

# =========================
# 4. ONE HOT
# =========================

# Encode les labels en vecteurs binaires de taille 10
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

# Softmax stable numériquement, compatible batch et vecteur seul
def softmax(o):
    exp_o = np.exp(o - np.max(o, axis=0, keepdims=True))
    return exp_o / np.sum(exp_o, axis=0, keepdims=True)

# =========================
# 7. ENTRAÎNEMENT
# =========================

# Descente de gradient par mini-batch
# Gradient de la cross-entropy : dA = (P - Y) @ X / batch, db = mean(P - Y)
lr = 0.1
batch_size = 64
losses = []

for epoch in range(20):
    idx = np.random.permutation(len(X_train))
    for i in range(0, len(X_train), batch_size):
        batch = idx[i:i+batch_size]
        x_b = X_train[batch].T       # (784, batch)
        y_b = Y_train[batch].T       # (10, batch)

        o = A @ x_b + b
        P = softmax(o)

        loss = -np.mean(np.sum(y_b * np.log(P + 1e-9), axis=0))
        losses.append(loss)

        dA = (P - y_b) @ x_b.T / batch_size
        db = np.mean(P - y_b, axis=1, keepdims=True)

        A -= lr * dA
        b -= lr * db

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | loss: {loss:.4f}")

# =========================
# 8. COURBE DE LOSS
# =========================
plt.plot(losses)
plt.title("Loss — Modèle linéaire")
plt.xlabel("Itération")
plt.ylabel("Cross-entropy")
plt.grid(True, alpha=0.3)
plt.show()

# =========================
# 9. ÉVALUATION
# =========================

# Calcule le taux d'erreur vectorisé sur un ensemble complet
def erreur(X, y):
    P = softmax(A @ X.T + b)
    preds = np.argmax(P, axis=0)
    return np.mean(preds != y) * 100

print(f"Erreur train : {erreur(X_train, y_train):.2f}%")
print(f"Erreur test  : {erreur(X_test,  y_test):.2f}%")