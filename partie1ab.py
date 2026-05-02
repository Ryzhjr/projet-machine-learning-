import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =============================================================================
# PARTIE 1 : MNIST — Modèle linéaire + MLP H=1 + MLP H=2
# Méthodes : Poly Ch.3 (softmax, cross-entropy, descente de gradient)
#            Poly Ch.4 (perceptron multi-couche, ReLU)
#            Poly Ch.5 (rétropropagation, Proposition 5.2)
# =============================================================================

# =========================
# 1. CHARGEMENT DES DONNÉES
# =========================
print("Chargement de MNIST...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Normalisation dans [0, 1] : diviser par 255 (valeur max d'un pixel)
# → améliore la descente de gradient (Poly §1.3.3 : préparation des données)
X = X.values / 255.0
y = y.values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42
)
print(f"Train : {X_train.shape[0]} images | Test : {X_test.shape[0]} images")

# =========================
# 2. ENCODAGE ONE-HOT
# =========================
# On crée la matrice Y (n × 10) telle que Y[i, k] = 1 si y_i = k, 0 sinon.
# Notation du sujet : y_i^(k) = 1 si l'image i appartient à la classe k.
def one_hot(y, nb_classes=10):
    Y = np.zeros((len(y), nb_classes))
    Y[np.arange(len(y)), y] = 1
    return Y

Y_train = one_hot(y_train)

# =============================================================================
# PARTIE A : MODÈLE LINÉAIRE (Poly Ch.3 + sujet §1.2.1)
# =============================================================================
# Scores (logits) : o_k = sum_j a_{k,j} x_j + a_{k,0}
# Sous forme matricielle : O = X @ A.T + b  (n × 10)
# Probabilités : P_k(x) = exp(o_k) / sum_j exp(o_j)   [softmax, Poly §4.6]
# Prédiction   : ŷ = argmax_k P_k(x)
#
# Fonction de coût : Log Loss (Poly §3.5)
# L(y, P) = -1/n sum_i sum_k y_i^(k) ln P_k(x_i)
#
# Gradient (Poly §3.4, adapté multi-classe) :
#   ∂L/∂A = (1/n) (P - Y)^T @ X    [taille 10 × 784]
#   ∂L/∂b = (1/n) sum_i (P_i - Y_i) [taille 1 × 10]
# =============================================================================

print("\n=== [A] MODÈLE LINÉAIRE (sans couche cachée) ===")

def softmax(O):
    """Softmax stable numériquement — convention ligne : O est (n, C)."""
    E = np.exp(O - np.max(O, axis=1, keepdims=True))
    return E / np.sum(E, axis=1, keepdims=True)

# Initialisation : petits poids aléatoires + biais nuls (Poly §2.3)
np.random.seed(42)
A_lin = np.random.randn(10, 784) * 0.01   # matrice 10 × 784
b_lin = np.zeros((1, 10))                 # biais 1 × 10

lr_lin      = 0.1
batch_size  = 64
epochs_lin  = 20
err_lin     = []   # taux d'erreur d'entraînement par époque

for epoch in range(epochs_lin):
    idx = np.random.permutation(len(X_train))
    for i in range(0, len(X_train), batch_size):
        batch   = idx[i:i+batch_size]
        X_b     = X_train[batch]          # (batch, 784)
        Y_b     = Y_train[batch]          # (batch, 10)

        # --- Propagation avant ---
        O       = X_b @ A_lin.T + b_lin  # (batch, 10)
        P       = softmax(O)              # (batch, 10)

        # --- Gradient de la Log Loss ---
        # ∂L/∂A = (1/n)(P - Y)^T @ X   [Poly §3.5 + règle de la chaîne §5.2]
        dA      = (P - Y_b).T @ X_b / len(X_b)   # (10, 784)
        db      = np.mean(P - Y_b, axis=0, keepdims=True)  # (1, 10)

        # --- Mise à jour des paramètres ---
        A_lin  -= lr_lin * dA
        b_lin  -= lr_lin * db

    # Taux d'erreur sur tout le train (pour suivi)
    O_full  = X_train @ A_lin.T + b_lin
    preds   = np.argmax(O_full, axis=1)
    err_lin.append(100 * np.mean(preds != y_train))
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:2d} | erreur train: {err_lin[-1]:.2f}%")

def predict_lin(X):
    return np.argmax(X @ A_lin.T + b_lin, axis=1)

err_lin_train = 100 * np.mean(predict_lin(X_train) != y_train)
err_lin_test  = 100 * np.mean(predict_lin(X_test)  != y_test)
print(f"  → Erreur finale — train: {err_lin_train:.2f}%  test: {err_lin_test:.2f}%")

# =============================================================================
# PARTIE B : MLP AVEC H COUCHES CACHÉES (Poly Ch.4 + Ch.5 + sujet §1.2.2)
# =============================================================================
# Architecture : [784, p1, ..., pH, 10]
# Couches cachées : activation ReLU  φ(x) = max(0, x)  [Poly §3.2.4]
# Couche de sortie : Softmax         [Poly §4.6]
# Fonction de coût : Log Loss        [Poly §3.5]
#
# Rétropropagation (Proposition 5.2 du Poly) :
#   δ_sortie = (P - Y) / n
#   Pour chaque couche h (de la sortie vers l'entrée) :
#     dW_h = A_{h-1}^T @ δ_h
#     db_h = sum(δ_h)
#     δ_{h-1} = δ_h @ W_h^T ⊙ ReLU'(Z_{h-1})
# =============================================================================

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    """Dérivée de ReLU : 1 si x > 0, 0 sinon."""
    return (x > 0).astype(float)

class MLP:
    """
    Perceptron multi-couche conforme au Poly (Ch.4-5).
    layers : ex [784, 256, 10] pour H=1, [784, 256, 128, 10] pour H=2.
    """
    def __init__(self, layers):
        self.params = {}
        self.num_layers = len(layers) - 1
        for i in range(self.num_layers):
            # Initialisation He pour ReLU :  σ = sqrt(2 / n_entrée)
            # → évite le vanishing/exploding gradient (Poly §3.4.3)
            self.params[f'W{i+1}'] = (
                np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i])
            )
            self.params[f'b{i+1}'] = np.zeros((1, layers[i+1]))

    def forward(self, X):
        """
        Propagation avant (Poly §5.2, Déf. 5.4 — phase 1).
        Retourne le dictionnaire des activations intermédiaires.
        """
        acts = {'A0': X}
        for i in range(1, self.num_layers + 1):
            # Pré-activation : o_q^h = sum_k a_qk^h z_k^{h-1} + biais  [Prop. 5.1]
            Z = acts[f'A{i-1}'] @ self.params[f'W{i}'] + self.params[f'b{i}']
            acts[f'Z{i}'] = Z
            # Activation : ReLU pour couches cachées, Softmax en sortie
            acts[f'A{i}'] = softmax(Z) if i == self.num_layers else relu(Z)
        return acts

    def backward(self, acts, Y):
        """
        Rétropropagation (Poly §5.2, Proposition 5.2).
        Calcule les gradients de la Log Loss par rapport à chaque paramètre.
        """
        grads = {}
        n = Y.shape[0]

        # Gradient en sortie : δ = (P - Y) / n  [dérivée de cross-entropy + softmax]
        dZ = (acts[f'A{self.num_layers}'] - Y) / n

        for i in range(self.num_layers, 0, -1):
            # ∂L/∂W_h = A_{h-1}^T @ δ_h   [Prop. 5.2]
            grads[f'dW{i}'] = acts[f'A{i-1}'].T @ dZ
            grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True)
            if i > 1:
                # Rétropropagation du gradient vers la couche précédente
                dA = dZ @ self.params[f'W{i}'].T
                dZ = dA * relu_deriv(acts[f'Z{i-1}'])   # ⊙ φ'(o^{h-1})
        return grads

    def update(self, grads, lr):
        """Descente de gradient : A ← A - δ ∂L/∂A  [Poly §2.3, Déf. 2.6]"""
        for i in range(1, self.num_layers + 1):
            self.params[f'W{i}'] -= lr * grads[f'dW{i}']
            self.params[f'b{i}'] -= lr * grads[f'db{i}']

    def predict(self, X):
        acts = self.forward(X)
        return np.argmax(acts[f'A{self.num_layers}'], axis=1)

def taux_erreur(model, X, y_true):
    return 100 * np.mean(model.predict(X) != y_true)

# ─────────────────────────────────────────────
# Entraînement MLP H=1 : [784 → 256 → 10]
# ─────────────────────────────────────────────
print("\n=== [B] MLP H=1 (784 → 256 → 10) ===")
mlp1   = MLP([784, 256, 10])
lr     = 0.01
batch_size = 64
epochs = 20
err_mlp1 = []

for epoch in range(epochs):
    idx = np.random.permutation(len(X_train))
    for i in range(0, len(X_train), batch_size):
        batch   = idx[i:i+batch_size]
        acts    = mlp1.forward(X_train[batch])
        grads   = mlp1.backward(acts, Y_train[batch])
        mlp1.update(grads, lr)

    e = taux_erreur(mlp1, X_train, y_train)
    err_mlp1.append(e)
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:2d} | erreur train: {e:.2f}%")

err_mlp1_train = taux_erreur(mlp1, X_train, y_train)
err_mlp1_test  = taux_erreur(mlp1, X_test,  y_test)
print(f"  → Erreur finale — train: {err_mlp1_train:.2f}%  test: {err_mlp1_test:.2f}%")

# ─────────────────────────────────────────────
# Entraînement MLP H=2 : [784 → 256 → 128 → 10]
# ─────────────────────────────────────────────
print("\n=== [C] MLP H=2 (784 → 256 → 128 → 10) ===")
mlp2     = MLP([784, 256, 128, 10])
err_mlp2 = []

for epoch in range(epochs):
    idx = np.random.permutation(len(X_train))
    for i in range(0, len(X_train), batch_size):
        batch   = idx[i:i+batch_size]
        acts    = mlp2.forward(X_train[batch])
        grads   = mlp2.backward(acts, Y_train[batch])
        mlp2.update(grads, lr)

    e = taux_erreur(mlp2, X_train, y_train)
    err_mlp2.append(e)
    if epoch % 5 == 0:
        print(f"  Epoch {epoch:2d} | erreur train: {e:.2f}%")

err_mlp2_train = taux_erreur(mlp2, X_train, y_train)
err_mlp2_test  = taux_erreur(mlp2, X_test,  y_test)
print(f"  → Erreur finale — train: {err_mlp2_train:.2f}%  test: {err_mlp2_test:.2f}%")

# =========================
# 3. COURBES D'ERREUR (3 modèles)
# =========================
plt.figure(figsize=(9, 5))
plt.plot(err_lin,  label='Linéaire (sans couche cachée)', color='royalblue')
plt.plot(err_mlp1, label='MLP H=1 (256 neurones)',        color='darkorange')
plt.plot(err_mlp2, label='MLP H=2 (256-128 neurones)',    color='green')
plt.xlabel("Époque")
plt.ylabel("Taux d'erreur entraînement (%)")
plt.title("MNIST — Courbes d'erreur : Linéaire vs MLP H=1 vs MLP H=2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mnist_courbes_erreur.pdf')
plt.show()

# =========================
# 4. TABLEAU RÉCAPITULATIF
# =========================
print("\n" + "=" * 55)
print(f"  {'Modèle':<28} {'Err. train':>10} {'Err. test':>10}")
print("=" * 55)
lignes = [
    ("Linéaire (sans couche cachée)", err_lin_train,  err_lin_test),
    ("MLP H=1 (256)",                 err_mlp1_train, err_mlp1_test),
    ("MLP H=2 (256-128)",             err_mlp2_train, err_mlp2_test),
]
for nom, tr, te in lignes:
    print(f"  {nom:<28} {tr:>9.2f}% {te:>9.2f}%")
print("=" * 55)

# Nombre de paramètres par modèle
nb_lin  = 10 * 784 + 10
nb_mlp1 = 784*256 + 256 + 256*10 + 10
nb_mlp2 = 784*256 + 256 + 256*128 + 128 + 128*10 + 10
print(f"\n  Paramètres — Linéaire: {nb_lin:,} | MLP H=1: {nb_mlp1:,} | MLP H=2: {nb_mlp2:,}")
print("\n  → Les couches cachées apprennent des représentations non linéaires")
print("    qui permettent de séparer des classes non linéairement séparables.")
print("  → L'ajout d'une 2e couche améliore l'expressivité du modèle (Ch.4).")

# =========================
# 5. MATRICE DE CONFUSION (meilleur modèle : MLP H=2)
# =========================
preds_test = mlp2.predict(X_test)
cm = confusion_matrix(y_test, preds_test)
disp = ConfusionMatrixDisplay(cm, display_labels=range(10))
disp.plot(cmap='Blues')
plt.title("Matrice de confusion — MLP H=2 (test)")
plt.tight_layout()
plt.savefig('mnist_confusion.pdf')
plt.show()

# =========================
# 6. EXEMPLES MAL CLASSÉS
# =========================
errors = [
    (X_test[i], y_test[i], p)
    for i, p in enumerate(preds_test)
    if p != y_test[i]
]
print(f"\nNombre d'erreurs (MLP H=2, test) : {len(errors)} / {len(y_test)}")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Exemples mal classés — MLP H=2\n(chiffres ambigus ou écritures atypiques)")
for ax, (img, vrai, pred) in zip(axes.flat, errors[:10]):
    ax.imshow(img.reshape(28, 28), cmap='gray')
    ax.set_title(f"Vrai:{vrai} → Prédit:{pred}", fontsize=9, color='red')
    ax.axis('off')
plt.tight_layout()
plt.savefig('mnist_mal_classes.pdf')
plt.show()

# =========================
# 7. PROJECTION t-SNE (Poly §1.2 — représentation en 2D)
# =========================
# Objectif : visualiser les représentations internes du MLP H=2
# (sortie de la 2e couche cachée, avant softmax)
# pour observer si les classes sont bien séparées dans l'espace latent.
print("\nProjection t-SNE des représentations internes (MLP H=2, 1000 points)...")
subset_idx  = np.random.choice(len(X_test), 1000, replace=False)
acts_test   = mlp2.forward(X_test[subset_idx])
repr_interne = acts_test['A2']   # sortie de la 2e couche cachée

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
proj = tsne.fit_transform(repr_interne)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    proj[:, 0], proj[:, 1],
    c=y_test[subset_idx], cmap='tab10', s=6, alpha=0.75
)
plt.colorbar(scatter, ticks=range(10), label='Classe (chiffre)')
plt.title("Projection t-SNE — Représentations internes MLP H=2 (test)\n"
          "Les clusters bien séparés montrent que le modèle a appris\n"
          "des représentations discriminantes pour chaque classe.")
plt.tight_layout()
plt.savefig('mnist_tsne.pdf')
plt.show()

# =========================
# 8. DISCUSSION FINALE (pour le rapport)
# =========================
print("\n" + "=" * 55)
print("  DISCUSSION")
print("=" * 55)
print("""
Limites du modèle linéaire :
- Frontière de décision linéaire : o_k = A_k·x + b_k
- Ne peut capturer des structures non linéaires dans les données.
- Equivalent à un seul perceptron par classe (Ch.3).

Apport des couches cachées (MLP) :
- Chaque couche apprend une représentation h de plus en plus abstraite.
- La profondeur réduit le nombre de neurones nécessaires (Th. approx. universelle, Ch.4).
- ReLU évite le vanishing gradient par rapport à la sigmoïde (Ch.3 §3.4.3).

Pourquoi MNIST est plus simple que CIFAR-10 :
- Images 28×28 niveaux de gris (vs 32×32 couleur).
- Peu de variabilité de fond, chiffres centrés.
- Séparation entre classes plus nette dans l'espace des pixels.
- Les MLP simples suffisent (~2% erreur), là où CIFAR-10 nécessite des CNN.

Rôle du nombre de paramètres :
- Linéaire :  {nb_lin:,} paramètres → trop peu expressif.
- MLP H=1 : {nb_mlp1:,} paramètres → bon compromis biais-variance.
- MLP H=2 : {nb_mlp2:,} paramètres → meilleure représentation, risque d'overfitting.
""".format(nb_lin=nb_lin, nb_mlp1=nb_mlp1, nb_mlp2=nb_mlp2))
