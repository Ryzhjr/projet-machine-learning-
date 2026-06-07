import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositif : {device}")

# IMG_SIZE = 128 conseillé. Sur CPU, 64 est nettement plus rapide.
IMG_SIZE = 128

# =========================
# 0. EMPLACEMENT ET TÉLÉCHARGEMENT DES DONNÉES
# =========================

# Dataset Kaggle "awsaf49/cbis-ddsm-breast-cancer-image-dataset" (images déjà en JPEG).
# Téléchargement automatique via l'API Kaggle si les fichiers sont absents
# (comme torchvision télécharge CIFAR-10).
# Prérequis : pip install kaggle  +  kaggle.json dans le dossier .kaggle de
#             l'utilisateur (Windows : %USERPROFILE%\.kaggle\ ; Linux/Mac : ~/.kaggle/)
#
# Structure attendue après décompression :
#   data/csv/mass_case_description_train_set.csv
#   data/csv/mass_case_description_test_set.csv
#   data/jpeg/<UID_serie>/1-xxx.jpg
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
CSV_TRAIN = os.path.join(DATA_DIR, "csv", "mass_case_description_train_set.csv")
CSV_TEST  = os.path.join(DATA_DIR, "csv", "mass_case_description_test_set.csv")
DATA_ROOT = os.path.join(DATA_DIR, "jpeg")

def telecharger_cbis_ddsm(data_dir):
    try:
        import kaggle
    except ImportError:
        raise ImportError("Package kaggle absent. Exécutez : pip install kaggle")
    print("Téléchargement de CBIS-DDSM depuis Kaggle (~6 Go)...")
    os.makedirs(data_dir, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "awsaf49/cbis-ddsm-breast-cancer-image-dataset",
        path=data_dir, unzip=True, quiet=False,
    )
    print("Téléchargement terminé.")

# Télécharge si le CSV d'entraînement est absent (équivalent de download=True)
if not os.path.exists(CSV_TRAIN):
    telecharger_cbis_ddsm(DATA_DIR)

# Le dataset Kaggle peut placer les fichiers dans un sous-dossier : on recherche
# les CSV et le dossier jpeg si les chemins par défaut ne correspondent pas.
def localiser(nom_fichier, racine):
    for r, _, fichiers in os.walk(racine):
        if nom_fichier in fichiers:
            return os.path.join(r, nom_fichier)
    return None

if not os.path.exists(CSV_TRAIN):
    trouve = localiser("mass_case_description_train_set.csv", DATA_DIR)
    if trouve:
        CSV_TRAIN = trouve
        CSV_TEST  = localiser("mass_case_description_test_set.csv", DATA_DIR)
        cand = os.path.join(os.path.dirname(os.path.dirname(CSV_TRAIN)), "jpeg")
        if os.path.isdir(cand):
            DATA_ROOT = cand

for f in [CSV_TRAIN, CSV_TEST, DATA_ROOT]:
    if not f or not os.path.exists(f):
        raise FileNotFoundError(f"\nFichier introuvable (le dataset) : {f}\n")

print(f"CSV train : {CSV_TRAIN}")
print(f"CSV test  : {CSV_TEST}")
print(f"Dossier images : {DATA_ROOT}")

# =========================
# 1. CHARGEMENT DU CSV ET DES ÉTIQUETTES
# =========================

# Contrairement à MNIST/CIFAR, l'étiquette n'est pas dans le nom du dossier :
# on la lit dans la colonne 'pathology'. Classification binaire :
#   BENIGN / BENIGN_WITHOUT_CALLBACK -> 0 (bénin)
#   MALIGNANT                        -> 1 (malin)
def charger_csv(path):
    df = pd.read_csv(path)
    benins = {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}
    df["label"] = df["pathology"].apply(lambda p: 0 if str(p).strip().upper() in benins else 1)
    # La colonne du chemin image s'appelle "image file path" dans ce dataset.
    col_path = next((c for c in df.columns if "image" in c.lower() and "path" in c.lower()), None)
    df = df.rename(columns={col_path: "image_path"})
    df = df[["image_path", "label", "pathology"]].dropna(subset=["image_path"]).reset_index(drop=True)
    return df

df_train = charger_csv(CSV_TRAIN)
df_test  = charger_csv(CSV_TEST)

print(f"\nTrain : {len(df_train)} cas")
print(f"  Bénins (0) : {(df_train['label']==0).sum()}")
print(f"  Malins (1) : {(df_train['label']==1).sum()}")
print(f"Test  : {len(df_test)} cas")
print(f"  Bénins (0) : {(df_test['label']==0).sum()}")
print(f"  Malins (1) : {(df_test['label']==1).sum()}")

# =========================
# 2. DISTRIBUTION DES CLASSES
# =========================

# On visualise le déséquilibre bénin/malin : c'est le défi central de ce dataset.
def plot_distribution(df, nom):
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Bénin (0)", "Malin (1)"], counts.values,
           color=["#4CAF50", "#F44336"], edgecolor="black")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontsize=11)
    ax.set_title(f"Distribution des classes — CBIS-DDSM ({nom})")
    ax.set_ylabel("Nombre d'images")
    plt.tight_layout()
    plt.savefig("cbis_distribution.png", dpi=100)
    plt.close()

plot_distribution(df_train, "train")

# =========================
# 3. LIEN IMAGE <-> ÉTIQUETTE (matching)
# =========================

# Les images JPEG sont rangées par dossier d'UID. On indexe ces dossiers une
# seule fois, puis on retrouve le bon JPEG à partir du chemin .dcm du CSV.
# Dans le chemin "..../UID_etude/UID_serie/000000.dcm", l'avant-dernier
# segment (UID_serie) correspond au nom du dossier dans jpeg/.
def construire_index(data_root):
    index = {}
    for dossier in os.listdir(data_root):
        chemin = os.path.join(data_root, dossier)
        if os.path.isdir(chemin):
            jpgs = sorted([f for f in os.listdir(chemin)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            if jpgs:
                index[dossier] = os.path.join(chemin, jpgs[0])
    print(f"  {len(index)} dossiers d'images indexés.")
    return index

def resoudre_chemin(image_path, index):
    segments = image_path.replace("\\", "/").split("/")
    # essai principal : avant-dernier segment (UID de la série)
    for s in (segments[-2:] if len(segments) >= 2 else segments):
        if s in index:
            return index[s]
    return None

class CBISDataset(Dataset):
    def __init__(self, df, index, transform):
        self.df = df.copy()
        self.df["chemin"] = self.df["image_path"].apply(lambda p: resoudre_chemin(p, index))
        manquantes = self.df["chemin"].isna().sum()
        if manquantes:
            print(f"  Attention : {manquantes} image(s) non retrouvée(s) — ignorée(s).")
        self.df = self.df.dropna(subset=["chemin"]).reset_index(drop=True)
        print(f"  {len(self.df)} images reliées à leur étiquette.")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["chemin"]).convert("L")   # niveaux de gris (1 canal)
        return self.transform(img), int(row["label"])

# =========================
# 4. PRÉTRAITEMENT
# =========================

# Redimensionnement (les originaux font jusqu'à 4000x3000) + normalisation des
# niveaux de gris dans [-1, 1]. Même prétraitement pour train et test.
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

index_images = construire_index(DATA_ROOT)
print("\nConstruction du jeu d'entraînement :")
train_set = CBISDataset(df_train, index_images, transform)
print("Construction du jeu de test :")
test_set  = CBISDataset(df_test,  index_images, transform)

loader_tr = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=0)
loader_te = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=0)

# =========================
# 5. EXEMPLES D'IMAGES
# =========================

def afficher_exemples(dataset, n=8):
    fig, axes = plt.subplots(2, n // 2, figsize=(13, 6))
    axes = axes.flatten()
    for i in range(n):
        img, label = dataset[i]
        axes[i].imshow(img.squeeze().numpy() * 0.5 + 0.5, cmap="gray")
        axes[i].set_title("MALIN" if label == 1 else "BÉNIN",
                          color="red" if label == 1 else "green", fontsize=10)
        axes[i].axis("off")
    plt.suptitle("Exemples — mammographies CBIS-DDSM", fontsize=13)
    plt.tight_layout()
    plt.savefig("cbis_exemples.png", dpi=100)
    plt.close()

afficher_exemples(train_set)

# =========================
# 6. ARCHITECTURE CNN
# =========================

# Même principe que la Partie 2b (Conv -> ReLU -> MaxPool), adapté :
#   - entrée 1 canal (niveaux de gris) au lieu de 3,
#   - sortie 2 neurones (bénin / malin) avec softmax via CrossEntropyLoss.
# Quatre Max-Pooling ramènent 128x128 -> 8x8, comme CIFAR finissait en 8x8x64.
class CNN_Mammo(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),   # 128 -> 64
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64  -> 32
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32  -> 16
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16  -> 8
        )
        # Taille après aplatissement, calculée automatiquement
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 2)     # 2 scores : bénin / malin
        )

    def forward(self, x):
        return self.classifier(self.features(x))

cnn = CNN_Mammo().to(device)
print(f"\nParamètres CNN : {sum(p.numel() for p in cnn.parameters()):,}")

# =========================
# 7. ENTRAÎNEMENT
# =========================

# Déséquilibre des classes : on pondère l'entropie croisée par l'inverse de la
# fréquence de chaque classe. Une erreur sur la classe rare (malin) "coûte"
# alors plus cher, sans changer l'architecture.
n0 = (df_train["label"] == 0).sum()
n1 = (df_train["label"] == 1).sum()
poids = torch.tensor([(n0 + n1) / (2 * n0), (n0 + n1) / (2 * n1)],
                     dtype=torch.float32).to(device)
print(f"Poids des classes (bénin, malin) : {poids.cpu().numpy().round(3)}")

crit = nn.CrossEntropyLoss(weight=poids)
opt  = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)

EPOCHS = 20
err_tr_hist, err_te_hist = [], []

def taux_erreur(loader):
    cnn.eval()
    faux, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = cnn(x.to(device)).argmax(1).cpu()
            faux  += (pred != y).sum().item()
            total += len(y)
    return 100 * faux / total

print(f"\n--- Entraînement CNN ({EPOCHS} époques) ---")
for ep in range(1, EPOCHS + 1):
    cnn.train()
    for x, y in loader_tr:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        crit(cnn(x), y).backward()
        opt.step()

    err_tr = taux_erreur(loader_tr)
    err_te = taux_erreur(loader_te)
    err_tr_hist.append(err_tr)
    err_te_hist.append(err_te)
    print(f"  Ep {ep:2d}/{EPOCHS} | err_train={err_tr:.1f}% | err_test={err_te:.1f}%")

# =========================
# 8. COURBES D'ERREUR
# =========================

plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS + 1), err_tr_hist, label="Train")
plt.plot(range(1, EPOCHS + 1), err_te_hist, label="Test")
plt.xlabel("Époque")
plt.ylabel("Erreur (%)")
plt.title("CNN — Courbes d'erreur CBIS-DDSM")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cbis_courbes.png", dpi=100)
plt.close()

# =========================
# 9. PRÉDICTIONS ET MATRICE DE CONFUSION
# =========================

# On collecte les vraies étiquettes, les prédictions et la probabilité de
# "malin" (2e sortie softmax) pour pouvoir faire varier le seuil ensuite.
cnn.eval()
y_true, y_pred, p_malin = [], [], []
with torch.no_grad():
    for x, y in loader_te:
        probs = torch.softmax(cnn(x.to(device)), dim=1)[:, 1].cpu().numpy()
        y_true.extend(y.numpy().tolist())
        p_malin.extend(probs.tolist())
        y_pred.extend((probs >= 0.5).astype(int).tolist())

y_true  = np.array(y_true)
y_pred  = np.array(y_pred)
p_malin = np.array(p_malin)

# Matrice de confusion calculée à la main (pas de bibliothèque) :
#   vrai \ prédit  | Bénin | Malin
#   Bénin          |  TN   |  FP
#   Malin          |  FN   |  TP
def confusion(yt, yp):
    TN = int(((yt == 0) & (yp == 0)).sum())
    FP = int(((yt == 0) & (yp == 1)).sum())
    FN = int(((yt == 1) & (yp == 0)).sum())
    TP = int(((yt == 1) & (yp == 1)).sum())
    return TN, FP, FN, TP

TN, FP, FN, TP = confusion(y_true, y_pred)
accuracy    = (TP + TN) / len(y_true)
sensibilite = TP / max(TP + FN, 1)   # = rappel sur les malins (cancers détectés)
specificite = TN / max(TN + FP, 1)   # bénins correctement classés
precision   = TP / max(TP + FP, 1)

print("\n" + "=" * 55)
print("  RÉSULTATS — CNN sur CBIS-DDSM (seuil 0.5)")
print("=" * 55)
print(f"  Accuracy    : {accuracy*100:.1f}%")
print(f"  Sensibilité : {sensibilite*100:.1f}%   (cancers détectés)")
print(f"  Spécificité : {specificite*100:.1f}%")
print(f"  Précision   : {precision*100:.1f}%")
print(f"  TN={TN}  FP={FP}  FN={FN}  TP={TP}")
print(f"  Faux négatifs (cancers manqués) : {FN}")
print("=" * 55)

# =========================
# 10. AFFICHAGE DE LA MATRICE DE CONFUSION
# =========================

cm = np.array([[TN, FP], [FN, TP]])
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Prédit Bénin", "Prédit Malin"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Vrai Bénin", "Vrai Malin"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=18,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
# On encadre la case des faux négatifs (malin prédit bénin) : l'erreur la plus grave.
ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, edgecolor="red", linewidth=3))
ax.text(0, 1.32, "faux négatifs", ha="center", color="red", fontsize=9)
ax.set_title("Matrice de confusion — CBIS-DDSM")
plt.tight_layout()
plt.savefig("cbis_confusion.png", dpi=100)
plt.close()

# =========================
# 11. EFFET DU SEUIL DE DÉCISION
# =========================

# En médecine, un faux négatif (cancer manqué) est plus grave qu'un faux positif
# (fausse alerte). En abaissant le seuil de décision, on classe "malin" plus
# facilement : la sensibilité augmente, au prix de la spécificité.
print("\nEffet du seuil de décision sur la sensibilité :")
print(f"  {'Seuil':>6} | {'Sensibilité':>11} | {'Spécificité':>11} | {'Faux nég.':>9}")
for seuil in [0.5, 0.4, 0.3]:
    yp = (p_malin >= seuil).astype(int)
    tn, fp, fn, tp = confusion(y_true, yp)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    print(f"  {seuil:>6.1f} | {sens*100:>10.1f}% | {spec*100:>10.1f}% | {fn:>9d}")

# =========================
# 12. EXEMPLES DE FAUX NÉGATIFS
# =========================

# Mammographies malignes que le modèle a classées bénignes : les erreurs
# les plus dangereuses, car elles retardent la prise en charge.
faux_neg = []
with torch.no_grad():
    for x, y in loader_te:
        probs = torch.softmax(cnn(x.to(device)), dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        for i in range(len(y)):
            if y[i].item() == 1 and preds[i] == 0:
                faux_neg.append((x[i].squeeze().numpy() * 0.5 + 0.5, probs[i]))
        if len(faux_neg) >= 6:
            break

if faux_neg:
    n = min(6, len(faux_neg))
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.flatten()
    for i in range(n):
        img, prob = faux_neg[i]
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Vrai : MALIN\nP(malin)={prob:.2f}", color="red", fontsize=9)
        axes[i].axis("off")
    for i in range(n, 6):
        axes[i].axis("off")
    plt.suptitle("Faux négatifs — cancers non détectés", fontsize=13, color="red")
    plt.tight_layout()
    plt.savefig("cbis_faux_negatifs.png", dpi=100)
    plt.close()
    print(f"\n{len(faux_neg)} faux négatif(s) illustré(s) dans cbis_faux_negatifs.png")
else:
    print("\nAucun faux négatif sur le jeu de test (seuil 0.5).")

print("\nPartie 3 terminée.")
