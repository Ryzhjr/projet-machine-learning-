import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# =========================
# 1. CHARGEMENT DES DONNÉES
# =========================
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X.values
y = y.values.astype(int)

# Normalisation
X = X / 255.0

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000)

# =========================
# 2. ONE HOT
# =========================
def one_hot(y):
    Y = np.zeros((len(y), 10))
    Y[np.arange(len(y)), y] = 1
    return Y

Y_train = one_hot(y_train)

# =========================
# 3. INITIALISATION
# =========================
W1 = np.random.randn(128, 784) * 0.01
b1 = np.zeros((128, 1))

W2 = np.random.randn(10, 128) * 0.01
b2 = np.zeros((10, 1))

# =========================
# 4. FONCTIONS
# =========================
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0)

def softmax(o):
    exp_o = np.exp(o - np.max(o, axis=0, keepdims=True))
    return exp_o / np.sum(exp_o, axis=0, keepdims=True)

# =========================
# 5. ENTRAÎNEMENT
# =========================
lr = 0.01
batch_size = 64

for epoch in range(5):
    for i in range(0, len(X_train), batch_size):
        
        x_batch = X_train[i:i+batch_size].T
        y_batch = Y_train[i:i+batch_size].T
        
        # FORWARD
        z1 = W1 @ x_batch + b1
        a1 = relu(z1)
        
        z2 = W2 @ a1 + b2
        P = softmax(z2)
        
        # BACKPROP
        dz2 = P - y_batch
        
        dW2 = dz2 @ a1.T / batch_size
        db2 = np.mean(dz2, axis=1, keepdims=True)
        
        da1 = W2.T @ dz2
        dz1 = da1 * relu_deriv(z1)
        
        dW1 = dz1 @ x_batch.T / batch_size
        db1 = np.mean(dz1, axis=1, keepdims=True)
        
        # UPDATE
        W1 -= lr * dW1
        b1 -= lr * db1
        
        W2 -= lr * dW2
        b2 -= lr * db2
    
    print(f"Epoch {epoch} terminé")

# =========================
# 6. TRAIN ACCURACY
# =========================
correct_train = 0

for i in range(len(X_train)):
    x = X_train[i].reshape(784,1)
    
    z1 = W1 @ x + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    
    pred = np.argmax(z2)
    
    if pred == y_train[i]:
        correct_train += 1

train_acc = correct_train / len(X_train)
print("Train accuracy:", train_acc)

# =========================
# 7. TEST + ERREURS
# =========================
correct_test = 0
errors = []

for i in range(len(X_test)):
    x = X_test[i].reshape(784,1)
    
    z1 = W1 @ x + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    
    pred = np.argmax(z2)
    
    if pred == y_test[i]:
        correct_test += 1
    else:
        errors.append((X_test[i], y_test[i], pred))

test_acc = correct_test / len(X_test)
print("Test accuracy:", test_acc)

# =========================
# 8. AFFICHER ERREURS
# =========================
print("\nExemples d'erreurs :")

for i in range(5):
    img, true, pred = errors[i]
    
    plt.imshow(img.reshape(28,28), cmap='gray')
    plt.title(f"Vrai: {true} / Prédit: {pred}")
    plt.show()

# =========================
# 9. PCA (VISUALISATION 2D)
# =========================
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_test)

plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y_test, cmap='tab10', s=2)
plt.title("Projection 2D des données (PCA)")
plt.show()