import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

# =========================
# 1. CHARGEMENT DES DONNÉES
# =========================
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X.values / 255.0
y = y.values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

# =========================
# 2. ONE HOT
# =========================

# Encode les labels en vecteurs binaires de taille 10
def one_hot(y):
    Y = np.zeros((len(y), 10))
    Y[np.arange(len(y)), y] = 1
    return Y

Y_train = one_hot(y_train)

# =========================
# 3. INITIALISATION
# =========================
W1 = np.random.randn(256, 784) * 0.01
b1 = np.zeros((256, 1))

W2 = np.random.randn(10, 256) * 0.01
b2 = np.zeros((10, 1))

# =========================
# 4. FONCTIONS
# =========================

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# Softmax stable numériquement, compatible batch
def softmax(o):
    exp_o = np.exp(o - np.max(o, axis=0, keepdims=True))
    return exp_o / np.sum(exp_o, axis=0, keepdims=True)

# =========================
# 5. ENTRAÎNEMENT
# =========================

# Backpropagation à la main sur réseau 784 -> 256 (ReLU) -> 10 (softmax)
lr = 0.01
batch_size = 64
losses = []

for epoch in range(20):
    idx = np.random.permutation(len(X_train))
    for i in range(0, len(X_train), batch_size):
        batch = idx[i:i+batch_size]
        x_b = X_train[batch].T
        y_b = Y_train[batch].T

        # Forward
        z1 = W1 @ x_b + b1
        a1 = relu(z1)
        z2 = W2 @ a1 + b2
        P  = softmax(z2)

        loss = -np.mean(np.sum(y_b * np.log(P + 1e-9), axis=0))
        losses.append(loss)

        # Backprop
        dz2 = (P - y_b) / batch_size
        dW2 = dz2 @ a1.T
        db2 = np.sum(dz2, axis=1, keepdims=True)

        dz1 = (W2.T @ dz2) * relu_deriv(z1)
        dW1 = dz1 @ x_b.T
        db1 = np.sum(dz1, axis=1, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    print(f"Epoch {epoch} | loss: {loss:.4f}")

# =========================
# 6. COURBE DE LOSS
# =========================
plt.plot(losses)
plt.title("Loss — MLP")
plt.xlabel("Itération")
plt.ylabel("Cross-entropy")
plt.grid(True, alpha=0.3)
plt.show()

# =========================
# 7. ÉVALUATION
# =========================

# Passe forward vectorisée pour évaluer le MLP sur un ensemble
def erreur(X, y):
    a1 = relu(W1 @ X.T + b1)
    P  = softmax(W2 @ a1 + b2)
    return np.mean(np.argmax(P, axis=0) != y) * 100

print(f"Erreur train : {erreur(X_train, y_train):.2f}%")
print(f"Erreur test  : {erreur(X_test,  y_test):.2f}%")

# =========================
# 8. AFFICHER ERREURS
# =========================

# Collecte les images mal classées pour analyse
errors = []
for i in range(len(X_test)):
    x = X_test[i].reshape(784, 1)
    a1 = relu(W1 @ x + b1)
    pred = np.argmax(W2 @ a1 + b2)
    if pred != y_test[i]:
        errors.append((X_test[i], y_test[i], pred))

print(f"\nNombre d'erreurs : {len(errors)} / {len(X_test)}")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Exemples mal classés — MLP")
for i, ax in enumerate(axes.flat):
    img, vrai, pred = errors[i]
    ax.imshow(img.reshape(28, 28), cmap='gray')
    ax.set_title(f"Vrai:{vrai} / Prédit:{pred}", fontsize=9, color='red')
    ax.axis('off')
plt.tight_layout()
plt.show()

# =========================
# 9. VISUALISATION 2D (t-SNE)
# =========================

# t-SNE sur 2000 exemples pour visualiser la séparation des classes
print("\nCalcul t-SNE (peut prendre ~1 min)...")
idx_viz = np.random.choice(len(X_test), 2000, replace=False)
X_2d = TSNE(n_components=2, random_state=42).fit_transform(X_test[idx_viz])

plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=y_test[idx_viz], cmap='tab10', s=5, alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label='Chiffre')
plt.title("Projection t-SNE — MNIST (test)")
plt.tight_layout()
plt.show()