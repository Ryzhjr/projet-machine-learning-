import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositif : {device}")

IMG_SIZE = 128   # 128 (CPU) ou 224 (GPU recommandé)

# =========================
# 0. TÉLÉCHARGEMENT AUTOMATIQUE
# =========================

# Contrairement à MNIST (OpenML) et CIFAR-10 (torchvision), CBIS-DDSM est un
# dataset médical hébergé sur Kaggle. L'API Kaggle permet de le télécharger
# automatiquement, exactement comme torchvision télécharge CIFAR-10.
#
# ÉTAPES (une seule fois) :
#   1. pip install kaggle
#   2. Aller sur https://www.kaggle.com -> Account -> "Create New Token"
#      -> cela télécharge kaggle.json
#   3. Placer kaggle.json dans : C:\Users\enzoc\.kaggle\kaggle.json
#      (créer le dossier .kaggle s'il n'existe pas)
#   Ensuite le script télécharge tout automatiquement comme CIFAR-10.

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
CSV_TRAIN = os.path.join(DATA_DIR, "csv", "mass_case_description_train_set.csv")
CSV_TEST  = os.path.join(DATA_DIR, "csv", "mass_case_description_test_set.csv")
DATA_ROOT = os.path.join(DATA_DIR, "jpeg")   # images JPEG dans le dataset Kaggle

def telecharger_cbis_ddsm(data_dir):
    """
    Télécharge CBIS-DDSM depuis Kaggle si les fichiers sont absents.
    Dataset : awsaf49/cbis-ddsm-breast-cancer-image-dataset
    Images déjà converties en JPEG — pas besoin de pydicom.
    """
    try:
        import kaggle
    except ImportError:
        raise ImportError(
            "\n\n❌ Le package kaggle n'est pas installé.\n"
            "   Exécutez : pip install kaggle\n"
            "   Puis créez votre token sur https://www.kaggle.com/settings\n"
            "   et placez kaggle.json dans C:\\Users\\enzoc\\.kaggle\\"
        )
    print("Téléchargement de CBIS-DDSM depuis Kaggle (~6 Go)...")
    print("(identique à torchvision qui télécharge CIFAR-10, mais plus lourd)")
    os.makedirs(data_dir, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "awsaf49/cbis-ddsm-breast-cancer-image-dataset",
        path=data_dir,
        unzip=True,
        quiet=False,
    )
    print("✓ Téléchargement terminé.")

# Téléchargement automatique si les fichiers sont absents (comme download=True dans CIFAR-10)
if not os.path.exists(CSV_TRAIN):
    telecharger_cbis_ddsm(DATA_DIR)

for f in [CSV_TRAIN, CSV_TEST]:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"\n\n❌ Fichier introuvable : {f}\n"
            "Vérifiez que kaggle.json est dans C:\\Users\\enzoc\\.kaggle\\"
        )
print(f"✓ CSV train : {CSV_TRAIN}")
print(f"✓ CSV test  : {CSV_TEST}")

# =========================
# 1. CHARGEMENT DU CSV
# =========================

# Le CSV CBIS-DDSM contient la colonne 'pathology' :
# BENIGN / BENIGN_WITHOUT_CALLBACK -> label 0
# MALIGNANT                        -> label 1
def charger_csv(path):
    df = pd.read_csv(path)
    BENIGNS = {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}
    df["label"] = df["pathology"].apply(lambda p: 0 if str(p).strip().upper() in BENIGNS else 1)
    # La colonne chemin image varie selon la version du CSV
    col_path = next((c for c in df.columns if "image" in c.lower() and "path" in c.lower()), None)
    df = df.rename(columns={col_path: "image_path"})
    df = df[["image_path", "label", "pathology"]].dropna(subset=["image_path"]).reset_index(drop=True)
    return df

df_train = charger_csv(CSV_TRAIN)
df_test  = charger_csv(CSV_TEST)

print(f"\nTrain : {len(df_train)} images")
print(f"  Bénins  (0) : {(df_train['label'] == 0).sum()}")
print(f"  Malins  (1) : {(df_train['label'] == 1).sum()}")
print(f"\nTest  : {len(df_test)} images")
print(f"  Bénins  (0) : {(df_test['label'] == 0).sum()}")
print(f"  Malins  (1) : {(df_test['label'] == 1).sum()}")

# =========================
# 2. DISTRIBUTION DES CLASSES
# =========================

# Visualise le déséquilibre de classes — problème central de ce dataset médical
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Distribution des classes — CBIS-DDSM", fontsize=13, fontweight='bold')

counts = df_train["label"].value_counts().sort_index()
couleurs = ["#4CAF50", "#F44336"]
labels_str = ["Bénin (0)", "Malin (1)"]

axes[0].bar(labels_str, counts.values, color=couleurs, edgecolor='white')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 3, str(v), ha='center', fontsize=11)
axes[0].set_title("Nombre d'images")
axes[0].set_ylabel("Effectif")

axes[1].pie(counts.values, labels=labels_str, colors=couleurs, autopct='%1.1f%%', startangle=90)
axes[1].set_title("Proportion")

plt.tight_layout()
plt.savefig('cbis_distribution.png', dpi=100)
plt.close()

# =========================
# 3. DATASET ET DATALOADERS
# =========================

# Construction d'un index unique : on scanne data/jpeg UNE SEULE FOIS
# puis on résout tous les chemins en mémoire — instantané
def construire_index(data_root):
    """
    Retourne un dict {uid_dossier: chemin_premier_jpg}
    CSV : Mass-Training/.../UID_PARENT/UID_IMAGE/000000.dcm
    JPEG : data/jpeg/UID_IMAGE/1-xxx.jpg  -> cle = UID_IMAGE
    """
    print("  Indexation des images (scan unique)...")
    index = {}
    for dossier in os.listdir(data_root):
        chemin_dossier = os.path.join(data_root, dossier)
        if os.path.isdir(chemin_dossier):
            fichiers = sorted([f for f in os.listdir(chemin_dossier)
                               if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            if fichiers:
                index[dossier] = os.path.join(chemin_dossier, fichiers[0])
    print(f"  ✓ {len(index)} dossiers indexés.")
    return index

def resoudre_chemin(image_path, index):
    # CSV : "Mass-Training_P_00001_LEFT_CC/UID1/UID2/000000.dcm"
    # UID2 = avant-dernier segment = nom du dossier dans jpeg/
    parties = image_path.replace("\\", "/").split("/")
    if len(parties) >= 2:
        uid = parties[-2]
        if uid in index:
            return index[uid]
    return None

class CBISDataset(Dataset):
    def __init__(self, df, data_root, index, transform=None):
        self.df = df.copy()
        self.df["path"] = self.df["image_path"].apply(lambda p: resoudre_chemin(p, index))
        manquantes = self.df["path"].isna().sum()
        if manquantes:
            print(f"  ⚠ {manquantes} image(s) introuvable(s) — ignorées.")
        self.df = self.df.dropna(subset=["path"]).reset_index(drop=True)
        print(f"  ✓ {len(self.df)} images trouvées.")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ext = os.path.splitext(row["path"])[1].lower()
        if ext == ".dcm":
            import pydicom
            dcm = pydicom.dcmread(row["path"])
            arr = dcm.pixel_array.astype(np.float32)
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            img = Image.fromarray(arr.astype(np.uint8)).convert("L")
        else:
            img = Image.open(row["path"]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"])

# Augmentation sur le train pour lutter contre l'overfitting
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

index_images = construire_index(DATA_ROOT)
train_set = CBISDataset(df_train, DATA_ROOT, index_images, transform_train)
test_set  = CBISDataset(df_test,  DATA_ROOT, index_images, transform_test)

# WeightedRandomSampler : surreprésente les malins pour compenser le déséquilibre
labels_tr = train_set.df["label"].values
class_counts  = np.bincount(labels_tr)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[l] for l in labels_tr]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

loader_tr = DataLoader(train_set, batch_size=16, sampler=sampler,  num_workers=0)
loader_te = DataLoader(test_set,  batch_size=16, shuffle=False,    num_workers=0)

print(f"\nDataLoaders prêts — train: {len(train_set)} | test: {len(test_set)}")

# =========================
# 4. EXEMPLES D'IMAGES
# =========================

# Affiche quelques mammographies avec leurs étiquettes
def afficher_exemples(loader, n=8):
    images, labels = next(iter(loader))
    n = min(n, len(images))
    fig, axes = plt.subplots(2, n//2, figsize=(14, 6))
    axes = axes.flatten()
    for i in range(n):
        img = images[i].squeeze().numpy() * 0.5 + 0.5
        label_str = "MALIN" if labels[i].item() == 1 else "BÉNIN"
        couleur   = "red" if labels[i].item() == 1 else "green"
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(label_str, color=couleur, fontsize=10, fontweight='bold')
        axes[i].axis('off')
    plt.suptitle("Exemples — CBIS-DDSM (mammographies)", fontsize=13)
    plt.tight_layout()
    plt.savefig('cbis_exemples.png', dpi=100)
    plt.close()

afficher_exemples(loader_tr)

# =========================
# 5. ARCHITECTURE CNN
# =========================

# CNN adapté de la Partie 2 : même principe (Conv → BN → ReLU → MaxPool)
# mais entrée en niveaux de gris (1 canal) et sortie binaire (1 neurone)
class CNN_Mammo(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),         # 224->112 / 128->64

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),         # 112->56 / 64->32

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),         # 56->28 / 32->16
        )
        # Calcul dynamique de la taille après flatten
        with torch.no_grad():
            dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
            flat  = self.features(dummy).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(flat, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)    # logit binaire — sigmoid appliqué dans la loss
        )

    def forward(self, x):
        return self.classifier(self.features(x))

cnn = CNN_Mammo().to(device)
nb_params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
print(f"\nArchitecture CNN_Mammo — {nb_params:,} paramètres entraînables")

# =========================
# 6. ENTRAÎNEMENT
# =========================

# pos_weight = nb_bénins / nb_malins : pénalise davantage les faux négatifs
# (un cancer non détecté est plus grave qu'une fausse alarme)
pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32).to(device)
print(f"pos_weight = {pos_weight.item():.2f}  (bénins={class_counts[0]}, malins={class_counts[1]})")

crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt  = optim.Adam(cnn.parameters(), lr=1e-3, weight_decay=1e-4)
sch  = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

EPOCHS = 25
hist = {'loss_tr': [], 'loss_te': [], 'acc_tr': [], 'acc_te': []}
meilleure_loss_te = float('inf')

print(f"\n--- Entraînement CNN ({EPOCHS} époques) ---")

for ep in range(1, EPOCHS + 1):
    # ── Phase train ──
    cnn.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader_tr:
        x = x.to(device)
        y = y.float().unsqueeze(1).to(device)
        opt.zero_grad()
        logits = cnn(x)
        loss   = crit(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        correct  += ((torch.sigmoid(logits) >= 0.5).float() == y).sum().item()
        total    += x.size(0)
    hist['loss_tr'].append(loss_sum / total)
    hist['acc_tr'].append(correct / total)

    # ── Phase test ──
    cnn.eval()
    loss_sum_te, correct_te, total_te = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader_te:
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device)
            logits = cnn(x)
            loss_sum_te += crit(logits, y).item() * x.size(0)
            correct_te  += ((torch.sigmoid(logits) >= 0.5).float() == y).sum().item()
            total_te    += x.size(0)
    hist['loss_te'].append(loss_sum_te / total_te)
    hist['acc_te'].append(correct_te  / total_te)

    sch.step(hist['loss_te'][-1])

    # Sauvegarde du meilleur modèle
    if hist['loss_te'][-1] < meilleure_loss_te:
        meilleure_loss_te = hist['loss_te'][-1]
        torch.save(cnn.state_dict(), 'meilleur_modele_mammo.pt')

    print(f"  Ep {ep:3d}/{EPOCHS} | "
          f"loss_train={hist['loss_tr'][-1]:.4f}  acc={hist['acc_tr'][-1]:.3f} | "
          f"loss_test={hist['loss_te'][-1]:.4f}  acc={hist['acc_te'][-1]:.3f}")

# Rechargement du meilleur modèle
cnn.load_state_dict(torch.load('meilleur_modele_mammo.pt', map_location=device))
print("✓ Meilleur modèle rechargé.")

# =========================
# 7. COURBES D'APPRENTISSAGE
# =========================

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Courbes d'apprentissage — CNN Mammographie", fontsize=13, fontweight='bold')
epochs_ax = range(1, EPOCHS + 1)

axes[0].plot(epochs_ax, hist['loss_tr'], label='Train', color='#2196F3')
axes[0].plot(epochs_ax, hist['loss_te'], label='Test',  color='#F44336')
axes[0].set_title("Fonction de coût (BCE)")
axes[0].set_xlabel("Époque")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_ax, hist['acc_tr'], label='Train', color='#2196F3')
axes[1].plot(epochs_ax, hist['acc_te'], label='Test',  color='#F44336')
axes[1].set_title("Exactitude (Accuracy)")
axes[1].set_xlabel("Époque")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cbis_courbes.png', dpi=100)
plt.close()

# =========================
# 8. PRÉDICTIONS ET MÉTRIQUES
# =========================

# Collecte toutes les prédictions sur le test set
cnn.eval()
y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for x, y in loader_te:
        logits = cnn(x.to(device))
        probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        preds  = (probs >= 0.5).astype(int)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

cm = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.ravel()
sensibilite = TP / max(TP + FN, 1)   # rappel malin
specificite = TN / max(TN + FP, 1)
precision   = TP / max(TP + FP, 1)
f1          = 2 * precision * sensibilite / max(precision + sensibilite, 1e-8)
accuracy    = (TP + TN) / len(y_true)
fn_rate     = FN / max(FN + TP, 1)   # cancers manqués

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc     = auc(fpr, tpr)

print("\n" + "="*55)
print("  RÉSULTATS — CNN sur CBIS-DDSM")
print("="*55)
print(f"  Accuracy       : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"  Sensibilité    : {sensibilite:.4f}  ← détecter les vrais cancers")
print(f"  Spécificité    : {specificite:.4f}")
print(f"  Précision      : {precision:.4f}")
print(f"  F1-score       : {f1:.4f}")
print(f"  AUC-ROC        : {roc_auc:.4f}")
print(f"  Taux FN        : {fn_rate:.4f}  ← cancers non détectés ⚠")
print("─"*55)
print(f"  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
print("="*55)

print("\n[Rapport sklearn]")
print(classification_report(y_true, y_pred, target_names=["Bénin", "Malin"]))

# =========================
# 9. MATRICE DE CONFUSION
# =========================

# La case FN (malin prédit bénin) est encadrée en rouge :
# c'est le type d'erreur le plus grave en diagnostic oncologique
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Prédit Bénin", "Prédit Malin"],
            yticklabels=["Vrai Bénin",  "Vrai Malin"],
            linewidths=0.5, linecolor='white',
            ax=ax, annot_kws={"size": 16})
ax.add_patch(plt.Rectangle((0, 1), 1, 1, fill=False, edgecolor='red', linewidth=3))
ax.text(0.5, 1.5, "FAUX NÉGATIFS\n(cancers manqués ⚠)",
        ha='center', va='center', color='red', fontsize=9, fontweight='bold')
ax.set_title("Matrice de confusion — CBIS-DDSM", fontsize=12)
plt.tight_layout()
plt.savefig('cbis_confusion.png', dpi=100)
plt.close()

# =========================
# 10. COURBE ROC
# =========================

# L'AUC-ROC mesure la qualité globale du discriminateur indépendamment du seuil
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Évaluation — Détection cancer du sein", fontsize=13, fontweight='bold')

axes[0].plot(fpr, tpr, color='#2196F3', lw=2, label=f"AUC = {roc_auc:.4f}")
axes[0].plot([0,1], [0,1], 'k--', lw=1, label="Classifieur aléatoire")
axes[0].fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
axes[0].set_xlabel("Taux de faux positifs (1 − Spécificité)")
axes[0].set_ylabel("Taux de vrais positifs (Sensibilité)")
axes[0].set_title("Courbe ROC")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Impact du seuil sur sensibilité / spécificité
from sklearn.metrics import precision_recall_curve
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
axes[1].plot(recall_vals, precision_vals, color='#4CAF50', lw=2)
axes[1].set_xlabel("Rappel (Sensibilité)")
axes[1].set_ylabel("Précision")
axes[1].set_title("Courbe Précision–Rappel")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cbis_roc.png', dpi=100)
plt.close()

# =========================
# 11. FAUX NÉGATIFS
# =========================

# Visualise les mammographies malignes que le modèle a manquées
# Ces erreurs sont les plus dangereuses : un cancer non détecté retarde le traitement
cnn.eval()
fn_images, fn_probs = [], []
with torch.no_grad():
    for x, y in loader_te:
        probs = torch.sigmoid(cnn(x.to(device))).squeeze(1).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        for i in range(len(y)):
            if y[i].item() == 1 and preds[i] == 0:
                fn_images.append(x[i].squeeze().numpy() * 0.5 + 0.5)
                fn_probs.append(probs[i])
        if len(fn_images) >= 8:
            break

if fn_images:
    n_show = min(8, len(fn_images))
    fig, axes = plt.subplots(2, n_show//2, figsize=(14, 6))
    axes = axes.flatten()
    for i in range(n_show):
        axes[i].imshow(fn_images[i], cmap='gray')
        axes[i].set_title(f"P(malin)={fn_probs[i]:.2f}\nVérité : MALIN", color='red', fontsize=9)
        axes[i].axis('off')
    plt.suptitle("Faux Négatifs — cancers non détectés ⚠", fontsize=13, color='red', fontweight='bold')
    plt.tight_layout()
    plt.savefig('cbis_faux_negatifs.png', dpi=100)
    plt.close()
    print(f"\n{len(fn_images)} faux négatifs trouvés (cancers non détectés).")
else:
    print("\nAucun faux négatif sur ce sous-ensemble de test.")

# =========================
# 12. RÉSUMÉ FINAL
# =========================

print("\n" + "="*55)
print("  RÉSUMÉ — Partie 3")
print("="*55)
print(f"  Accuracy finale (test) : {hist['acc_te'][-1]*100:.2f}%")
print(f"  Meilleure accuracy     : {max(hist['acc_te'])*100:.2f}%")
print(f"  AUC-ROC                : {roc_auc:.4f}")
print()
print("  Points clés pour la soutenance :")
print("  1. Déséquilibre de classes géré par WeightedRandomSampler")
print("     et pos_weight dans BCEWithLogitsLoss.")
print("  2. Un FN (malin manqué) est plus grave qu'un FP (fausse alarme).")
print("     Abaisser le seuil (ex: 0.3) maximise la sensibilité.")
print(f"  3. Redimensionnement {IMG_SIZE}×{IMG_SIZE} : perte d'information")
print("     par rapport aux images originales (jusqu'à 4000×3000 px).")
print("  4. La Batch Normalization et le Dropout limitent l'overfitting")
print("     sur un dataset de taille limitée comparé à CIFAR-10.")
print("="*55)
print("\nPartie 3 terminée.")
