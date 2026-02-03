import pandas as pd
import numpy as np
import os
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix, precision_score, classification_report

# --- 0. CONFIGURARE CĂI RELATIVE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
PATH_BASE = os.path.abspath(os.path.join(current_dir, "..", ".."))

PATH_DATA = os.path.join(PATH_BASE, "data", "test")
PATH_MODELS = os.path.join(PATH_BASE, "models")
PATH_SAVE_FINAL = os.path.join(PATH_BASE, "results", "etapa6")
PATH_CONFIG = os.path.join(PATH_BASE, "config")

os.makedirs(PATH_SAVE_FINAL, exist_ok=True)
os.makedirs(PATH_CONFIG, exist_ok=True)

def run_evaluation():
    # --- 2. ÎNCĂRCARE DATE TEST ---
    p_path = os.path.join(PATH_DATA, "pneumonie", "pneumonie_test.csv")
    t_path = os.path.join(PATH_DATA, "tuberculoza", "tuberculoza_test.csv")
    
    if not os.path.exists(p_path) or not os.path.exists(t_path):
        print(f"❌ Datele de test nu au fost găsite în: {PATH_DATA}")
        return

    df_p = pd.read_csv(p_path)
    df_t = pd.read_csv(t_path)
    df_test = pd.concat([df_p, df_t], ignore_index=True)
    
    X_test = df_test.drop('Diagnosis', axis=1).values.astype('float32')
    y_test = df_test['Diagnosis'].values.astype('float32')

    # --- 3. ÎNCĂRCARE MODEL (.keras) ---
    model_path = os.path.join(PATH_MODELS, "optimized_model.keras")
    if not os.path.exists(model_path):
        print(f"❌ Modelul optimizat nu a fost găsit la {model_path}")
        return

    model = tf.keras.models.load_model(model_path)

    # --- 4. CALCUL METRICI DETALIATE ---
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    acc = np.mean(y_pred == y_test)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    
    tbc_recall = cm[1,1] / (cm[1,1] + cm[1,0])
    pneu_recall = cm[0,0] / (cm[0,0] + cm[0,1])
    
    tbc_precision = precision_score(y_test, y_pred, pos_label=1)
    pneu_precision = precision_score(y_test, y_pred, pos_label=0)
    
    total_samples = len(y_test)
    fn_rate = cm[1,0] / total_samples
    fp_rate = cm[0,1] / total_samples

    # --- 5. ANALIZĂ EXEMPLE (ADĂUGAT) ---
    # Selectăm 3 corecte din fiecare clasă
    correct_idx = np.where(y_pred == y_test)[0]
    pneu_correct = [int(i) for i in correct_idx if y_test[i] == 0][:3]
    tbc_correct = [int(i) for i in correct_idx if y_test[i] == 1][:3]
    
    correct_list = []
    for idx in (pneu_correct + tbc_correct):
        correct_list.append({
            "sample_index": idx,
            "real": "TBC" if y_test[idx] == 1 else "Pneumonie",
            "confidence": round(float(y_pred_prob[idx][0]), 4),
            "key_symptoms": {"Q1": float(X_test[idx][0]), "Q16_Sange": float(X_test[idx][15])}
        })

    # Selectăm 5 erori
    errors_idx = np.where(y_pred != y_test)[0]
    error_list = []
    for i in range(min(5, len(errors_idx))):
        idx = int(errors_idx[i])
        error_list.append({
            "sample_index": idx,
            "real": "TBC" if y_test[idx] == 1 else "Pneumonie",
            "pred": "Pneumonie" if y_test[idx] == 1 else "TBC",
            "confidence": round(float(y_pred_prob[idx][0]), 4),
            "analysis": "Overlap simptomatic."
        })

    # --- 6. SALVARE REZULTATE ---
    # Salvare YAML
    config_data = {
        "model_metadata": {"name": "TBC_Pneumo_Balanced_MLP", "version": "1.0_Etapa6"},
        "architecture": {"input_features": 20, "hidden_layers": [64, 64, 64], "dropout": 0.2},
        "inference": {"threshold": 0.5}
    }
    with open(os.path.join(PATH_CONFIG, "optimized_config.yaml"), "w") as y_file:
        yaml.dump(config_data, y_file, default_flow_style=False)

    # Salvare JSON (Metrici + Analiză Exemple)
    final_output = {
        "metrics": {
            "accuracy": round(float(acc), 4),
            "tbc_recall": round(float(tbc_recall), 4),
            "pneu_recall": round(float(pneu_recall), 4)
        },
        "correct_samples_analysis": correct_list,
        "error_analysis": error_list
    }
    
    with open(os.path.join(PATH_SAVE_FINAL, "final_test_report.json"), "w") as f:
        json.dump(final_output, f, indent=4)

    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', xticklabels=['Pneumonie', 'TBC'], yticklabels=['Pneumonie', 'TBC'])
    plt.title('Confusion Matrix Finală (Etapa 6)')
    plt.savefig(os.path.join(PATH_SAVE_FINAL, "final_confusion_matrix.png"))
    plt.close()
    
    print(f"✅ Evaluare completă! Raportul și exemplele au fost salvate în: {PATH_SAVE_FINAL}")

if __name__ == "__main__":
    run_evaluation()
