import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['avion','automobile','oiseau','chat','cerf','chien','grenouille','cheval','bateau','camion']

# =========================
# 1. CHARGEMENT CIFAR-10
# =========================

# Augmentation sur le train pour réduire l'overfitting
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform_train)
test_set  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

loader_tr = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=0)
loader_te = DataLoader(test_set,  batch_size=128, shuffle=False, num_workers=0)

# =========================
# 2. CONVOLUTION MANUELLE
# =========================

# Applique un filtre 3x3 sur une image en niveaux de gris avec zero-padding
# Formule du cours : m'(u,v) = sum_{u',v'} K_{u',v'} * m_{u+u'-2, v+v'-2} + biais
def convolution(image, K, biais=0):
    H, W = image.shape
    img_pad = np.pad(image, 1, mode='constant')
    sortie = np.zeros((H, W))
    for u in range(H):
        for v in range(W):
            sortie[u, v] = np.sum(K * img_pad[u:u+3, v:v+3]) + biais
    return sortie

# Normalise une image filtrée dans [0,1] pour l'affichage
def normaliser(img):
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)

# Récupère la première image de chat du dataset (classe 3)
def get_chat(dataset):
    for img, label in dataset:
        if label == 3:
            img = img * torch.tensor([0.2470,0.2435,0.2616]).view(3,1,1) \
                      + torch.tensor([0.4914,0.4822,0.4465]).view(3,1,1)
            img = np.clip(img.numpy(), 0, 1)
            return 0.299*img[0] + 0.587*img[1] + 0.114*img[2]

# =========================
# 3. FILTRES K1..K6 SUR IMAGE DE CHAT
# =========================

filtres = {
    'K1 - Flou':       (1/9)*np.ones((3,3)),
    'K2 - Nettete':    np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
    'K3 - Contours V': np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]]),
    'K4 - Contours H': np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),
    'K5 - Sobel H':    np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
    'K6 - Sobel D':    np.array([[-2,-1,0],[-1,1,1],[0,1,2]]),
}

chat = get_chat(test_set)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Effet des filtres de convolution (image de chat)", fontsize=13)

axes[0][0].imshow(chat, cmap='gray')
axes[0][0].set_title("Original")
axes[0][0].axis('off')

positions = [(0,1),(0,2),(0,3),(1,0),(1,1),(1,2)]
for (r,c), (nom, K) in zip(positions, filtres.items()):
    axes[r][c].imshow(normaliser(convolution(chat, K)), cmap='gray')
    axes[r][c].set_title(nom, fontsize=9)
    axes[r][c].axis('off')

axes[1][3].axis('off')
plt.tight_layout()
plt.savefig('cifar_filtres.png', dpi=100)
plt.show()

# =========================
# 4. ARCHITECTURE CNN
# =========================

# CNN conforme au sujet :
# (32,32,3) -> Conv64 -> Conv64 -> Pool -> Conv64 -> Pool -> Conv64 -> Flatten -> Dense(10)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64*8*8, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# =========================
# 5. ENTRAÎNEMENT CNN
# =========================

# Entraîne le CNN avec Adam + scheduler cosinus
cnn  = CNN().to(device)
opt  = optim.Adam(cnn.parameters(), lr=1e-3, weight_decay=1e-4)
sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
crit = nn.CrossEntropyLoss()

print(f"Parametres CNN : {sum(p.numel() for p in cnn.parameters()):,}")
print("\n--- Entrainement CNN (20 epoques) ---")

hist_cnn = {'err_tr': [], 'err_te': []}

for ep in range(1, 21):
    cnn.train()
    for x, y in loader_tr:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        crit(cnn(x), y).backward()
        opt.step()
    sch.step()

    cnn.eval()
    correct_tr, correct_te = 0, 0
    with torch.no_grad():
        for x, y in loader_tr:
            correct_tr += (cnn(x.to(device)).argmax(1) == y.to(device)).sum().item()
        for x, y in loader_te:
            correct_te += (cnn(x.to(device)).argmax(1) == y.to(device)).sum().item()

    err_tr = 100 * (1 - correct_tr / len(train_set))
    err_te = 100 * (1 - correct_te / len(test_set))
    hist_cnn['err_tr'].append(err_tr)
    hist_cnn['err_te'].append(err_te)

    print(f"  Ep {ep}/20 | err_train={err_tr:.1f}% | err_test={err_te:.1f}%")

# =========================
# 6. COURBES CNN
# =========================

# Trace les courbes d'erreur du CNN pour visualiser l'overfitting éventuel
plt.figure(figsize=(8, 4))
plt.plot(hist_cnn['err_tr'], label='Train')
plt.plot(hist_cnn['err_te'], label='Test')
plt.xlabel("Epoque")
plt.ylabel("Erreur (%)")
plt.title("CNN — Courbes d'erreur CIFAR-10")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cnn_courbes.png', dpi=100)
plt.show()

# =========================
# 7. IMAGES MAL CLASSÉES
# =========================

# Affiche des exemples d'images mal classées avec leur vraie classe et la prédiction
cnn.eval()
errors = []
with torch.no_grad():
    for x, y in loader_te:
        preds = cnn(x.to(device)).argmax(1).cpu()
        for i in range(len(y)):
            if preds[i] != y[i] and len(errors) < 10:
                img = x[i] * torch.tensor([0.2470,0.2435,0.2616]).view(3,1,1) \
                           + torch.tensor([0.4914,0.4822,0.4465]).view(3,1,1)
                errors.append((img.permute(1,2,0).numpy(), y[i].item(), preds[i].item()))
        if len(errors) == 10:
            break

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Exemples mal classes — CNN CIFAR-10")
for i, ax in enumerate(axes.flat):
    img, vrai, pred = errors[i]
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title(f"Vrai:{CLASSES[vrai]}\nPredit:{CLASSES[pred]}", fontsize=8, color='red')
    ax.axis('off')
plt.tight_layout()
plt.savefig('cnn_erreurs.png', dpi=100)
plt.show()

# =========================
# 8. RÉSUMÉ FINAL
# =========================

print(f"\n  Erreur finale CNN — train: {hist_cnn['err_tr'][-1]:.2f}%  test: {hist_cnn['err_te'][-1]:.2f}%")
print(f"  Meilleure erreur test     : {min(hist_cnn['err_te']):.2f}%")
print("\nPartie 2B terminee.")