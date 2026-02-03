import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

PATH_BASE = os.path.abspath(os.path.join(current_dir, "../../"))

PATH_OUT = os.path.join(PATH_BASE, "results", "etapa6final")
FILE_NAME = "example_predictions_final.png"
FULL_SAVE_PATH = os.path.join(PATH_OUT, FILE_NAME)

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT, exist_ok=True)

# --- 2. DATELE  ---
data = [
    # 6 Exemple Corecte (Succes)
    {"ID": 0, "Status": "Succes", "Real": "Pneu", "Pred": "Pneu", "Conf": 0.0077, "Sange": 0.0, "Greutate": 0.25, "Febra": 0.75},
    {"ID": 1, "Status": "Succes", "Real": "Pneu", "Pred": "Pneu", "Conf": 0.0275, "Sange": 1.0, "Greutate": 0.0, "Febra": 0.5},
    {"ID": 2, "Status": "Succes", "Real": "Pneu", "Pred": "Pneu", "Conf": 0.0047, "Sange": 0.25, "Greutate": 0.25, "Febra": 0.75},
    {"ID": 2251, "Status": "Succes", "Real": "TBC", "Pred": "TBC", "Conf": 0.7987, "Sange": 0.0, "Greutate": 0.75, "Febra": 0.75},
    {"ID": 2252, "Status": "Succes", "Real": "TBC", "Pred": "TBC", "Conf": 0.8226, "Sange": 0.25, "Greutate": 0.75, "Febra": 0.25},
    {"ID": 2253, "Status": "Succes", "Real": "TBC", "Pred": "TBC", "Conf": 0.9653, "Sange": 0.5, "Greutate": 0.5, "Febra": 0.25},
    # 5 Exemple Greșite (Eroare)
    {"ID": 10, "Status": "Eroare", "Real": "Pneu", "Pred": "TBC", "Conf": 0.5607, "Sange": 1.0, "Greutate": 0.5, "Febra": 0.5},
    {"ID": 15, "Status": "Eroare", "Real": "Pneu", "Pred": "TBC", "Conf": 0.5736, "Sange": 0.75, "Greutate": 0.5, "Febra": 0.5},
    {"ID": 31, "Status": "Eroare", "Real": "Pneu", "Pred": "TBC", "Conf": 0.6144, "Sange": 1.0, "Greutate": 0.0, "Febra": 0.75},
    {"ID": 71, "Status": "Eroare", "Real": "Pneu", "Pred": "TBC", "Conf": 0.5419, "Sange": 0.25, "Greutate": 1.0, "Febra": 0.0},
    {"ID": 72, "Status": "Eroare", "Real": "Pneu", "Pred": "TBC", "Conf": 0.5772, "Sange": 0.5, "Greutate": 0.5, "Febra": 0.0}
]

df = pd.DataFrame(data)

# --- 3. VIZUALIZARE (GRID DE ANALIZĂ) ---
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(22, 10))
plt.subplots_adjust(hspace=0.5, wspace=0.3)
axes = axes.flatten()

for i, row in df.iterrows():
    ax = axes[i]
    color = '#2ecc71' if row['Status'] == "Succes" else '#e74c3c'
    
    ax.bar(['Confidence'], [row['Conf']], color=color, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.2)
    
    ax.set_title(f"ID: {row['ID']} ({row['Status']})", fontsize=10, fontweight='bold', color=color)
    desc = f"R: {row['Real']} | P: {row['Pred']}\nSânge: {row['Sange']}\nGreut: {row['Greutate']}"
    ax.set_xlabel(desc, fontsize=9)

axes[-1].axis('off') 
plt.suptitle("Analiza Granulară a Predicțiilor (Etapa 6)", fontsize=20, y=0.98)

# --- 4. SALVARE ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(FULL_SAVE_PATH, dpi=300, bbox_inches='tight')

if os.path.exists(FULL_SAVE_PATH):
    print(f"✅ SUCCES! Vizualul a fost salvat la cale relativă.")
    print(f"📂 Locație finală: {os.path.relpath(FULL_SAVE_PATH, os.getcwd())}")
else:
    print("❌ EROARE la salvare.")

plt.show()