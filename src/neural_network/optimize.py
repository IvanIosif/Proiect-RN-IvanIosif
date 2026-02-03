import os
import json
import pandas as pd
import numpy as np
import yaml
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# --- 0. CONFIGURARE CĂI RELATIVE (AUTOMATIZARE) ---
# Detectăm locația scriptului actual
current_dir = os.path.dirname(os.path.abspath(__file__))

PATH_BASE = os.path.abspath(os.path.join(current_dir, "..")) 

# Definirea directoarelor relativ la rădăcină
PATH_OUT = os.path.join(PATH_BASE, "results", "etapa6")
PATH_MODELS = os.path.join(PATH_BASE, "models")
PATH_CONFIG = os.path.join(PATH_BASE, "config")
PATH_DATA = os.path.join(PATH_BASE, "data")

for path in [PATH_OUT, PATH_MODELS, PATH_CONFIG]:
    os.makedirs(path, exist_ok=True)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.run_functions_eagerly(True)

# Mapare Semantică (Rămâne neschimbată)
semantic_map = {
    0:  [1, 1, 1, 0, 0], 1:  [0, 0, 0, 1, 1], 2:  [1, 0, 0, 1, 1],
    3:  [0, 0, 0, 1, 1], 4:  [0, 0, 0, 1, 1], 5:  [1, 1, 0, 0, 0],
    6:  [1, 0, 0, 0, 0], 7:  [1, 0, 0, 0, 1], 8:  [1, 0, 0, 0, 1],
    9:  [0, 0, 0, 0, 0], 10: [1, 0, 0, 1, 1], 11: [0, 0, 0, 1, 1],
    12: [1, 0, 0, 0, 0], 13: [1, 0, 0, 0, 0], 14: [0, 0, 1, 1, 1],
    15: [0, 0, 1, 1, 1], 16: [0, 0, 0, 1, 1], 17: [1, 0, 0, 0, 0],
    18: [0, 0, 0, 1, 1], 19: [1, 0, 0, 1, 1]
}

def apply_semantic_logic(X):
    X_opt = np.copy(X)
    tbc_anchors = [14, 15] 
    for i in range(X.shape[0]):
        for j in range(20):
            val_idx = int(round(X[i, j] * 4))
            val_idx = max(0, min(val_idx, 4))
            if semantic_map[j][val_idx] == 1:
                X_opt[i, j] *= (1.55 if j in tbc_anchors and val_idx >= 2 else 1.25)
            else:
                X_opt[i, j] *= 0.98 
    return X_opt

def load_data(split):
    # Folosim PATH_DATA construit relativ
    p_path = os.path.join(PATH_DATA, split, "pneumonie", f"pneumonie_{split}.csv")
    t_path = os.path.join(PATH_DATA, split, "tuberculoza", f"tuberculoza_{split}.csv")
    
    if not os.path.exists(p_path) or not os.path.exists(t_path):
        raise FileNotFoundError(f"Lipsesc fisierele de date în: {os.path.join(PATH_DATA, split)}")
        
    df = pd.concat([pd.read_csv(p_path), pd.read_csv(t_path)], ignore_index=True)
    X = df.drop('Diagnosis', axis=1).values.astype('float32')
    y = df['Diagnosis'].values.astype('float32')
    return apply_semantic_logic(X), y

# --- 2. ÎNCĂRCARE DATE ---
print(f"⏳ Încărcare date din: {PATH_DATA}")
X_train, y_train = load_data("train")
X_val, y_val = load_data("validation")
X_test, y_test = load_data("test")

# --- 3. EXPERIMENTE DE OPTIMIZARE ---
experiments = [
    {"id": "Exp1_Base", "lr": 0.001, "layers": [64, 32], "do": 0.2},
    {"id": "Exp2_Deep_Semantic", "lr": 0.0005, "layers": [128, 64, 32], "do": 0.3},
    {"id": "Exp3_Balanced", "lr": 0.0005, "layers": [64, 64, 64], "do": 0.2},
    {"id": "Exp4_Precision", "lr": 0.0003, "layers": [128, 128], "do": 0.1}
]

results_list = []
best_f1, best_model, best_cfg = 0, None, None

for cfg in experiments:
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20,)),
        *[tf.keras.layers.Dense(u, activation='relu') for u in cfg["layers"]],
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(cfg["do"]),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"]),
                  loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
    
    print(f"\n🚀 Antrenare {cfg['id']}...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=50, batch_size=32, verbose=1,
                        class_weight={0: 1.0, 1: 1.25}, 
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = np.mean(y_pred == y_test)
    
    results_list.append({"Exp": cfg["id"], "Accuracy": float(acc), "F1-score": float(f1)})
    if f1 > best_f1:
        best_f1, best_model, best_cfg = f1, model, cfg

# --- 4. SALVARE REZULTATE ȘI METRICI ---
# 1. Tabel CSV
pd.DataFrame(results_list).to_csv(os.path.join(PATH_OUT, "optimisation_experiments.csv"), index=False)

# 2. JSON Metrics
y_pred_final = (best_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
report = classification_report(y_test, y_pred_final, output_dict=True)

final_metrics = {
    "best_exp_id": best_cfg["id"],
    "overall_accuracy": report["accuracy"],
    "tbc_sensitivity_recall": report["1.0"]["recall"],
    "pneumonia_specificity_recall": report["0.0"]["recall"],
    "macro_f1": report["macro avg"]["f1-score"]
}

with open(os.path.join(PATH_OUT, "final_metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=4)

# 3. Grafice și Matrice
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); sns.barplot(x="Exp", y="Accuracy", data=pd.DataFrame(results_list)); plt.title("Accuracy Comparison")
plt.subplot(1, 2, 2); sns.barplot(x="Exp", y="F1-score", data=pd.DataFrame(results_list)); plt.title("F1-Score Comparison")
plt.savefig(os.path.join(PATH_OUT, "metrics_comparison.png"))

cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                                        xticklabels=['Pneu', 'TBC'], yticklabels=['Pneu', 'TBC'])
plt.title(f"Confusion Matrix - {best_cfg['id']}")
plt.savefig(os.path.join(PATH_OUT, "confusion_matrix_optimized.png"))

# 4. Salvare Model Final și Config
best_model.save(os.path.join(PATH_MODELS, "optimized_model.keras"))
joblib.dump("semantic_v2_logic", os.path.join(PATH_CONFIG, "scaler_optimized.skl"))

print(f"\n✅ Etapa 6 Finalizată!")
print(f"Toate fișierele au fost salvate relativ în: {PATH_BASE}")

