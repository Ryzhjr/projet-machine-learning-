import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. CHARGEMENT CIFAR-10
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
test_set  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

CLASSES = ['avion','automobile','oiseau','chat','cerf','chien','grenouille','cheval','bateau','camion']

print(f"Train : {len(train_set)} images | Test : {len(test_set)} images")

# =========================
# 2. VISUALISATION
# =========================

# Affiche quelques images par classe pour vérifier le chargement
def afficher_exemples(dataset, classes, n=5):
    buckets = {i: [] for i in range(10)}
    for img, label in dataset:
        if len(buckets[label]) < n:
            buckets[label].append(img)
        if all(len(v) == n for v in buckets.values()):
            break

    fig, axes = plt.subplots(10, n, figsize=(n*2, 22))
    for c in range(10):
        for j in range(n):
            ax = axes[c][j]
            ax.imshow(buckets[c][j].permute(1,2,0).numpy() * 0.5 + 0.5)
            if j == 0:
                ax.set_ylabel(classes[c], fontsize=10, rotation=0, labelpad=50, va='center')
            ax.axis('off')
    plt.suptitle("Exemples CIFAR-10", fontsize=13)
    plt.tight_layout()
    plt.savefig('cifar_exemples.png', dpi=100)
    plt.show()

afficher_exemples(train_set, CLASSES)

# =========================
# 2B. COULEUR VS NIVEAUX DE GRIS
# =========================

# Compare visuellement les images couleur et leur version en niveaux de gris
def plot_gris_vs_couleur(dataset, classes, n=5):
    exemples = []
    for img, label in dataset:
        exemples.append((img, label))
        if len(exemples) == n:
            break

    fig, axes = plt.subplots(2, n, figsize=(n*2.5, 6))
    fig.suptitle("CIFAR-10 : couleur vs niveaux de gris", fontsize=13, fontweight='bold')
    for j, (img, label) in enumerate(exemples):
        img_show = img * torch.tensor([0.2470,0.2435,0.2616]).view(3,1,1) \
                       + torch.tensor([0.4914,0.4822,0.4465]).view(3,1,1)
        axes[0][j].imshow(img_show.permute(1,2,0).numpy().clip(0,1))
        axes[0][j].set_title(classes[label], fontsize=9)
        axes[0][j].axis('off')
        gray = (0.299*img_show[0] + 0.587*img_show[1] + 0.114*img_show[2]).numpy()
        axes[1][j].imshow(gray, cmap='gray')
        axes[1][j].axis('off')
    axes[0][0].set_ylabel('Couleur', fontsize=10)
    axes[1][0].set_ylabel('Niveaux de gris', fontsize=10)
    plt.tight_layout()
    plt.savefig('cifar_gris_vs_couleur.png', dpi=100)
    plt.show()

plot_gris_vs_couleur(test_set, CLASSES)

# =========================
# 3. CONVERSION NIVEAUX DE GRIS
# =========================

# Convertit les images en niveaux de gris selon la formule du cours
# x_j = 0.299*R + 0.587*G + 0.114*B puis aplatit en vecteur 1024
def to_gray(dataset):
    X, y = [], []
    for img, label in dataset:
        img = img * torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1) \
                  + torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
        gray = 0.299*img[0] + 0.587*img[1] + 0.114*img[2]
        X.append(gray.view(-1))
        y.append(label)
    return torch.stack(X), torch.tensor(y)

print("Conversion en niveaux de gris...")
X_tr_gray, y_tr = to_gray(train_set)
X_te_gray, y_te = to_gray(test_set)

X_tr_gray = (X_tr_gray - X_tr_gray.mean()) / X_tr_gray.std()
X_te_gray = (X_te_gray - X_te_gray.mean()) / X_te_gray.std()

# =========================
# 4. VERSION COULEUR (3072)
# =========================

# Aplatit directement les images couleur en vecteur 3072 (32*32*3)
def to_flat(dataset):
    X, y = [], []
    for img, label in dataset:
        X.append(img.view(-1))
        y.append(label)
    return torch.stack(X), torch.tensor(y)

print("Aplatissement couleur...")
X_tr_col, _ = to_flat(train_set)
X_te_col, _ = to_flat(test_set)

# =========================
# 5. MODÈLES LINÉAIRE ET MLP
# =========================

# Régression softmax simple : une seule couche linéaire
class Lineaire(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 10)
    def forward(self, x):
        return self.fc(x)

# MLP avec deux couches cachées ReLU
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.net(x)

# =========================
# 6. BOUCLE D'ENTRAÎNEMENT
# =========================

# Entraîne un modèle et retourne les taux d'erreur train/test
def entrainer(model, X_tr, y_tr, X_te, y_te, epochs=30, lr=1e-3, bs=256):
    model.to(device)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
    opt  = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    err_tr_hist, err_te_hist = [], []

    for ep in range(1, epochs+1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            err_tr = 1 - (model(X_tr.to(device)).argmax(1) == y_tr.to(device)).float().mean().item()
            err_te = 1 - (model(X_te.to(device)).argmax(1) == y_te.to(device)).float().mean().item()
        err_tr_hist.append(err_tr * 100)
        err_te_hist.append(err_te * 100)

        if ep % 10 == 0:
            print(f"  Ep {ep}/{epochs} | err_train={err_tr*100:.1f}% | err_test={err_te*100:.1f}%")

    return err_tr_hist, err_te_hist

# =========================
# 7. ENTRAÎNEMENTS
# =========================

print("\n--- Linéaire niveaux de gris ---")
lin_gray = Lineaire(1024)
h_lin_gray = entrainer(lin_gray, X_tr_gray, y_tr, X_te_gray, y_te)

print("\n--- MLP niveaux de gris ---")
mlp_gray = MLP(1024)
h_mlp_gray = entrainer(mlp_gray, X_tr_gray, y_tr, X_te_gray, y_te)

print("\n--- Linéaire couleur ---")
lin_col = Lineaire(3072)
h_lin_col = entrainer(lin_col, X_tr_col, y_tr, X_te_col, y_te)

print("\n--- MLP couleur ---")
mlp_col = MLP(3072)
h_mlp_col = entrainer(mlp_col, X_tr_col, y_tr, X_te_col, y_te)

# =========================
# 8. COURBES D'ERREUR
# =========================

# Trace les courbes de taux d'erreur test pour les 4 modèles
def plot_erreurs(hists, noms):
    plt.figure(figsize=(10, 5))
    for (_, h_te), nom in zip(hists, noms):
        plt.plot(h_te, label=nom)
    plt.xlabel("Époque")
    plt.ylabel("Erreur test (%)")
    plt.title("Comparaison des modèles — CIFAR-10")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cifar_courbes.png', dpi=100)
    plt.show()

hists = [h_lin_gray, h_mlp_gray, h_lin_col, h_mlp_col]
noms  = ['Linéaire gris', 'MLP gris', 'Linéaire couleur', 'MLP couleur']
plot_erreurs(hists, noms)

# =========================
# 9. TABLEAU RÉSULTATS
# =========================

print("\n" + "="*60)
print(f"  {'Modèle':<30} {'Err. train':>10} {'Err. test':>10}")
print("="*60)
for (h_tr, h_te), nom in zip(hists, noms):
    print(f"  {nom:<30} {h_tr[-1]:>9.2f}% {h_te[-1]:>9.2f}%")

print("\n  Littérature scientifique (pour comparaison) :")
litterature = [
    ("Conv. Deep Belief (2010)",      21.1),
    ("Maxout Networks (2013)",          9.38),
    ("Densely Connected CNN (2016)",    3.46),
    ("Vision Transformer (2021)",       0.50),
]
for nom, err in litterature:
    print(f"  {nom:<35} {err:.2f}%")
print("="*60)
print("\n-> Les MLP simples ne capturent pas les motifs spatiaux.")
print("-> Les CNNs sont necessaires pour atteindre de bonnes performances.")