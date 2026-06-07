import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# =============================================================================
# PARTIE 1 : MNIST
# Modèle linéaire, MLP H=1, MLP H=2
# =============================================================================


# =============================================================================
# 1. Chargement et préparation des données
# =============================================================================

print("Chargement de MNIST...")

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

# Les images sont en 28 x 28 pixels, donc 784 valeurs par image.
# On ramène les valeurs des pixels entre 0 et 1.
X = X.values / 255.0
y = y.values.astype(int)

print("Données chargées :", X.shape)


def separation_train_test(X, y, n_test=10000, seed=42):
    """
    Sépare les données en un ensemble d'entraînement et un ensemble de test.
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(X))

    indices_test = indices[:n_test]
    indices_train = indices[n_test:]

    X_train = X[indices_train]
    X_test = X[indices_test]
    y_train = y[indices_train]
    y_test = y[indices_test]

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = separation_train_test(X, y, n_test=10000)

print(f"Train : {X_train.shape[0]} images")
print(f"Test  : {X_test.shape[0]} images")


# =============================================================================
# 2. Encodage des étiquettes
# =============================================================================

def one_hot(y, nb_classes=10):
    """
    Transforme les étiquettes en matrice.
    Exemple : si le chiffre est 3, la colonne 3 vaut 1 et les autres valent 0.
    """
    Y = np.zeros((len(y), nb_classes))
    Y[np.arange(len(y)), y] = 1
    return Y


Y_train = one_hot(y_train, nb_classes=10)
Y_test = one_hot(y_test, nb_classes=10)


# =============================================================================
# 3. Fonctions utiles
# =============================================================================

def softmax(O):
    """
    Transforme les scores en probabilités.
    """
    O_stable = O - np.max(O, axis=1, keepdims=True)
    exp_O = np.exp(O_stable)
    return exp_O / np.sum(exp_O, axis=1, keepdims=True)


def log_loss(Y, P):
    """
    Fonction coût utilisée pour la classification multi-classe.
    """
    eps = 1e-12
    return -np.mean(np.sum(Y * np.log(P + eps), axis=1))


def taux_erreur_prediction(y_true, y_pred):
    """
    Calcule le taux d'erreur en pourcentage.
    """
    return 100 * np.mean(y_true != y_pred)


# =============================================================================
# 4. Modèle linéaire multi-classe
# =============================================================================

print("\n=== [A] Modèle linéaire multi-classe ===")

# Le modèle calcule les scores :
# o = A x + b
# Ici, A est une matrice de taille 10 x 784.
# Chaque ligne correspond à une classe.

np.random.seed(42)

A_lin = np.random.randn(10, 784) * 0.01
b_lin = np.zeros((1, 10))

delta_lin = 0.1
L = 64
epochs_lin = 20

err_lin_train_courbe = []
loss_lin_courbe = []


def predict_lineaire(X, A, b):
    """
    Renvoie la classe qui a le plus grand score.
    """
    O = X @ A.T + b
    return np.argmax(O, axis=1)


for epoch in range(epochs_lin):

    indices = np.random.permutation(len(X_train))

    for i in range(0, len(X_train), L):
        batch = indices[i:i + L]

        X_b = X_train[batch]
        Y_b = Y_train[batch]

        # Calcul des scores puis des probabilités
        O = X_b @ A_lin.T + b_lin
        P = softmax(O)

        # Calcul des gradients
        n_b = X_b.shape[0]
        dA = (P - Y_b).T @ X_b / n_b
        db = np.mean(P - Y_b, axis=0, keepdims=True)

        # Mise à jour des paramètres
        A_lin = A_lin - delta_lin * dA
        b_lin = b_lin - delta_lin * db

    # Suivi de l'évolution du modèle
    O_train = X_train @ A_lin.T + b_lin
    P_train = softmax(O_train)
    y_pred_train = np.argmax(P_train, axis=1)

    err_train = taux_erreur_prediction(y_train, y_pred_train)
    loss_train = log_loss(Y_train, P_train)

    err_lin_train_courbe.append(err_train)
    loss_lin_courbe.append(loss_train)

    if epoch % 5 == 0:
        print(f"Epoch {epoch:2d} | erreur train : {err_train:.2f}% | loss : {loss_train:.4f}")


err_lin_train = taux_erreur_prediction(y_train, predict_lineaire(X_train, A_lin, b_lin))
err_lin_test = taux_erreur_prediction(y_test, predict_lineaire(X_test, A_lin, b_lin))

print(f"Erreur finale modèle linéaire — train : {err_lin_train:.2f}% | test : {err_lin_test:.2f}%")


# =============================================================================
# 5. Perceptron multi-couche
# =============================================================================

def relu(x):
    """
    Fonction d'activation ReLU.
    """
    return np.maximum(0, x)


def relu_derivee(x):
    """
    Dérivée de ReLU.
    """
    return (x > 0).astype(float)


class MLP:
    """
    Réseau de neurones avec une ou plusieurs couches cachées.

    Exemples :
    H=1 : [784, 256, 10]
    H=2 : [784, 256, 128, 10]
    """

    def __init__(self, couches, seed=42):
        np.random.seed(seed)

        self.couches = couches
        self.nb_couches = len(couches) - 1
        self.params = {}

        for h in range(1, self.nb_couches + 1):
            taille_entree = couches[h - 1]
            taille_sortie = couches[h]

            self.params[f"W{h}"] = np.random.randn(taille_entree, taille_sortie) * np.sqrt(2 / taille_entree)
            self.params[f"b{h}"] = np.zeros((1, taille_sortie))

    def forward(self, X):
        """
        Propagation avant.
        """
        cache = {}
        cache["A0"] = X

        for h in range(1, self.nb_couches + 1):

            W = self.params[f"W{h}"]
            b = self.params[f"b{h}"]

            Z = cache[f"A{h-1}"] @ W + b
            cache[f"Z{h}"] = Z

            if h == self.nb_couches:
                cache[f"A{h}"] = softmax(Z)
            else:
                cache[f"A{h}"] = relu(Z)

        return cache

    def backward(self, cache, Y):
        """
        Rétropropagation.
        """
        gradients = {}
        n = Y.shape[0]

        dZ = (cache[f"A{self.nb_couches}"] - Y) / n

        for h in range(self.nb_couches, 0, -1):

            A_precedent = cache[f"A{h-1}"]

            gradients[f"dW{h}"] = A_precedent.T @ dZ
            gradients[f"db{h}"] = np.sum(dZ, axis=0, keepdims=True)

            if h > 1:
                W = self.params[f"W{h}"]
                dA_precedent = dZ @ W.T
                dZ = dA_precedent * relu_derivee(cache[f"Z{h-1}"])

        return gradients

    def update(self, gradients, delta):
        """
        Mise à jour des poids et des biais.
        """
        for h in range(1, self.nb_couches + 1):
            self.params[f"W{h}"] = self.params[f"W{h}"] - delta * gradients[f"dW{h}"]
            self.params[f"b{h}"] = self.params[f"b{h}"] - delta * gradients[f"db{h}"]

    def predict(self, X):
        """
        Renvoie la classe prédite.
        """
        cache = self.forward(X)
        P = cache[f"A{self.nb_couches}"]
        return np.argmax(P, axis=1)


def entrainer_mlp(modele, X_train, Y_train, y_train, epochs=20, L=64, delta=0.01):
    """
    Entraîne un MLP avec des mini-lots.
    """
    erreurs = []
    pertes = []

    for epoch in range(epochs):

        indices = np.random.permutation(len(X_train))

        for i in range(0, len(X_train), L):
            batch = indices[i:i + L]

            X_b = X_train[batch]
            Y_b = Y_train[batch]

            cache = modele.forward(X_b)
            gradients = modele.backward(cache, Y_b)
            modele.update(gradients, delta)

        cache_train = modele.forward(X_train)
        P_train = cache_train[f"A{modele.nb_couches}"]

        y_pred_train = np.argmax(P_train, axis=1)

        err = taux_erreur_prediction(y_train, y_pred_train)
        loss = log_loss(Y_train, P_train)

        erreurs.append(err)
        pertes.append(loss)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | erreur train : {err:.2f}% | loss : {loss:.4f}")

    return erreurs, pertes


# =============================================================================
# 6. MLP H = 1
# =============================================================================

print("\n=== [B] MLP H=1 : 784 -> 256 -> 10 ===")

mlp1 = MLP([784, 256, 10], seed=42)

delta_mlp = 0.01
epochs_mlp = 20
L = 64

err_mlp1_train_courbe, loss_mlp1_courbe = entrainer_mlp(
    mlp1,
    X_train,
    Y_train,
    y_train,
    epochs=epochs_mlp,
    L=L,
    delta=delta_mlp
)

err_mlp1_train = taux_erreur_prediction(y_train, mlp1.predict(X_train))
err_mlp1_test = taux_erreur_prediction(y_test, mlp1.predict(X_test))

print(f"Erreur finale MLP H=1 — train : {err_mlp1_train:.2f}% | test : {err_mlp1_test:.2f}%")


# =============================================================================
# 7. MLP H = 2
# =============================================================================

print("\n=== [C] MLP H=2 : 784 -> 256 -> 128 -> 10 ===")

mlp2 = MLP([784, 256, 128, 10], seed=42)

err_mlp2_train_courbe, loss_mlp2_courbe = entrainer_mlp(
    mlp2,
    X_train,
    Y_train,
    y_train,
    epochs=epochs_mlp,
    L=L,
    delta=delta_mlp
)

err_mlp2_train = taux_erreur_prediction(y_train, mlp2.predict(X_train))
err_mlp2_test = taux_erreur_prediction(y_test, mlp2.predict(X_test))

print(f"Erreur finale MLP H=2 — train : {err_mlp2_train:.2f}% | test : {err_mlp2_test:.2f}%")


# =============================================================================
# 8. Courbes d'erreur
# =============================================================================

plt.figure(figsize=(9, 5))
plt.plot(err_lin_train_courbe, label="Modèle linéaire")
plt.plot(err_mlp1_train_courbe, label="MLP H=1")
plt.plot(err_mlp2_train_courbe, label="MLP H=2")
plt.xlabel("Époque")
plt.ylabel("Taux d'erreur entraînement (%)")
plt.title("MNIST — Courbes d'erreur")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mnist_courbes_erreur.pdf")
plt.show()


# =============================================================================
# 9. Courbes de Log Loss
# =============================================================================

plt.figure(figsize=(9, 5))
plt.plot(loss_lin_courbe, label="Modèle linéaire")
plt.plot(loss_mlp1_courbe, label="MLP H=1")
plt.plot(loss_mlp2_courbe, label="MLP H=2")
plt.xlabel("Époque")
plt.ylabel("Log Loss")
plt.title("MNIST — Évolution de la fonction coût")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mnist_courbes_log_loss.pdf")
plt.show()


# =============================================================================
# 10. Résultats
# =============================================================================

print("\n" + "=" * 65)
print(f"{'Modèle':<35} {'Erreur train':>14} {'Erreur test':>14}")
print("=" * 65)

resultats = [
    ("Modèle linéaire", err_lin_train, err_lin_test),
    ("MLP H=1 : 784-256-10", err_mlp1_train, err_mlp1_test),
    ("MLP H=2 : 784-256-128-10", err_mlp2_train, err_mlp2_test)
]

for nom, err_train, err_test in resultats:
    print(f"{nom:<35} {err_train:>13.2f}% {err_test:>13.2f}%")

print("=" * 65)

nb_lin = 10 * 784 + 10
nb_mlp1 = 784 * 256 + 256 + 256 * 10 + 10
nb_mlp2 = 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10

print("\nNombre de paramètres :")
print(f"Modèle linéaire : {nb_lin}")
print(f"MLP H=1         : {nb_mlp1}")
print(f"MLP H=2         : {nb_mlp2}")


# =============================================================================
# 11. Matrice de confusion
# =============================================================================

def matrice_confusion(y_true, y_pred, nb_classes=10):
    """
    M[i,j] contient le nombre d'images de classe i prédites comme classe j.
    """
    M = np.zeros((nb_classes, nb_classes), dtype=int)

    for vrai, pred in zip(y_true, y_pred):
        M[vrai, pred] += 1

    return M


def afficher_matrice_confusion(M, titre="Matrice de confusion"):
    """
    Affiche la matrice de confusion.
    """
    plt.figure(figsize=(8, 7))
    plt.imshow(M)
    plt.colorbar(label="Nombre d'images")

    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.title(titre)

    plt.xticks(range(10))
    plt.yticks(range(10))

    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("mnist_matrice_confusion.pdf")
    plt.show()


preds_test_mlp2 = mlp2.predict(X_test)
M_conf = matrice_confusion(y_test, preds_test_mlp2, nb_classes=10)

afficher_matrice_confusion(M_conf, titre="Matrice de confusion — MLP H=2")


# =============================================================================
# 12. Exemples mal classés
# =============================================================================

erreurs = []

for i in range(len(X_test)):
    if preds_test_mlp2[i] != y_test[i]:
        erreurs.append((X_test[i], y_test[i], preds_test_mlp2[i]))

print(f"\nNombre d'erreurs MLP H=2 sur le test : {len(erreurs)} / {len(y_test)}")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Exemples mal classés — MLP H=2")

for ax, (image, vrai, pred) in zip(axes.flat, erreurs[:10]):
    ax.imshow(image.reshape(28, 28), cmap="gray")
    ax.set_title(f"Vrai : {vrai} | Prédit : {pred}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("mnist_exemples_mal_classes.pdf")
plt.show()


# =============================================================================
# 13. ACP sur les données brutes et sur les représentations du MLP
# =============================================================================

def acp_2d(X):
    """
    Réalise une ACP en deux dimensions.
    """

    X_centre = X - np.mean(X, axis=0)

    C = (X_centre.T @ X_centre) / X_centre.shape[0]

    valeurs_propres, vecteurs_propres = np.linalg.eigh(C)

    ordre = np.argsort(valeurs_propres)[::-1]
    valeurs_propres = valeurs_propres[ordre]
    vecteurs_propres = vecteurs_propres[:, ordre]

    axes_principaux = vecteurs_propres[:, :2]
    X_proj = X_centre @ axes_principaux

    variance_expliquee = valeurs_propres[:2] / np.sum(valeurs_propres)

    return X_proj, variance_expliquee


print("\nACP sur les données brutes et sur les représentations du MLP...")

np.random.seed(42)
indices_acp = np.random.choice(len(X_test), 1000, replace=False)

X_acp_source = X_test[indices_acp]
y_acp_source = y_test[indices_acp]


# -----------------------------------------------------------------------------
# ACP 1 : pixels bruts x
# -----------------------------------------------------------------------------

X_proj_brut, variance_brut = acp_2d(X_acp_source)


# -----------------------------------------------------------------------------
# ACP 2 : représentation z1 du MLP H=1
# -----------------------------------------------------------------------------

cache_mlp1 = mlp1.forward(X_acp_source)
Z1_mlp1 = cache_mlp1["A1"]

Z1_proj, variance_z1 = acp_2d(Z1_mlp1)


# -----------------------------------------------------------------------------
# ACP 3 : représentation z2 du MLP H=2
# -----------------------------------------------------------------------------

cache_mlp2 = mlp2.forward(X_acp_source)

# Pour le MLP H=2 :
# A1 correspond à la première représentation cachée
# A2 correspond à la deuxième représentation cachée
Z2_mlp2 = cache_mlp2["A2"]

Z2_proj, variance_z2 = acp_2d(Z2_mlp2)


# -----------------------------------------------------------------------------
# Affichage séparé : données brutes
# -----------------------------------------------------------------------------

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_proj_brut[:, 0],
    X_proj_brut[:, 1],
    c=y_acp_source,
    cmap="tab10",
    s=8,
    alpha=0.75
)

plt.colorbar(scatter, ticks=range(10), label="Classe réelle")
plt.xlabel(f"Axe principal 1 ({variance_brut[0] * 100:.2f}% variance)")
plt.ylabel(f"Axe principal 2 ({variance_brut[1] * 100:.2f}% variance)")
plt.title("ACP — Pixels bruts")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mnist_acp_pixels_bruts.pdf")
plt.show()


# -----------------------------------------------------------------------------
# Affichage séparé : représentation z1
# -----------------------------------------------------------------------------

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    Z1_proj[:, 0],
    Z1_proj[:, 1],
    c=y_acp_source,
    cmap="tab10",
    s=8,
    alpha=0.75
)

plt.colorbar(scatter, ticks=range(10), label="Classe réelle")
plt.xlabel(f"Axe principal 1 ({variance_z1[0] * 100:.2f}% variance)")
plt.ylabel(f"Axe principal 2 ({variance_z1[1] * 100:.2f}% variance)")
plt.title("ACP — Représentation z¹ du MLP H=1")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mnist_acp_representation_z1.pdf")
plt.show()


# -----------------------------------------------------------------------------
# Affichage séparé : représentation z2
# -----------------------------------------------------------------------------

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    Z2_proj[:, 0],
    Z2_proj[:, 1],
    c=y_acp_source,
    cmap="tab10",
    s=8,
    alpha=0.75
)

plt.colorbar(scatter, ticks=range(10), label="Classe réelle")
plt.xlabel(f"Axe principal 1 ({variance_z2[0] * 100:.2f}% variance)")
plt.ylabel(f"Axe principal 2 ({variance_z2[1] * 100:.2f}% variance)")
plt.title("ACP — Représentation z² du MLP H=2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mnist_acp_representation_z2.pdf")
plt.show()


# -----------------------------------------------------------------------------
# Affichage côte à côte pour comparer
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

scatter0 = axes[0].scatter(
    X_proj_brut[:, 0],
    X_proj_brut[:, 1],
    c=y_acp_source,
    cmap="tab10",
    s=8,
    alpha=0.75
)
axes[0].set_title("Pixels bruts")
axes[0].set_xlabel(f"Axe 1 ({variance_brut[0] * 100:.2f}%)")
axes[0].set_ylabel(f"Axe 2 ({variance_brut[1] * 100:.2f}%)")
axes[0].grid(True, alpha=0.3)

scatter1 = axes[1].scatter(
    Z1_proj[:, 0],
    Z1_proj[:, 1],
    c=y_acp_source,
    cmap="tab10",
    s=8,
    alpha=0.75
)
axes[1].set_title("Après 1 couche cachée : z¹")
axes[1].set_xlabel(f"Axe 1 ({variance_z1[0] * 100:.2f}%)")
axes[1].set_ylabel(f"Axe 2 ({variance_z1[1] * 100:.2f}%)")
axes[1].grid(True, alpha=0.3)

scatter2 = axes[2].scatter(
    Z2_proj[:, 0],
    Z2_proj[:, 1],
    c=y_acp_source,
    cmap="tab10",
    s=8,
    alpha=0.75
)
axes[2].set_title("Après 2 couches cachées : z²")
axes[2].set_xlabel(f"Axe 1 ({variance_z2[0] * 100:.2f}%)")
axes[2].set_ylabel(f"Axe 2 ({variance_z2[1] * 100:.2f}%)")
axes[2].grid(True, alpha=0.3)

fig.colorbar(scatter2, ax=axes, ticks=range(10), label="Classe réelle")
fig.suptitle("Comparaison des représentations : x → z¹ → z²", fontsize=14)

plt.tight_layout()
plt.savefig("mnist_acp_comparaison_x_z1_z2.pdf")
plt.show()


print("\nVariance expliquée par les deux premiers axes :")
print(f"Pixels bruts : {(variance_brut[0] + variance_brut[1]) * 100:.2f}%")
print(f"z1 MLP H=1   : {(variance_z1[0] + variance_z1[1]) * 100:.2f}%")
print(f"z2 MLP H=2   : {(variance_z2[0] + variance_z2[1]) * 100:.2f}%")
