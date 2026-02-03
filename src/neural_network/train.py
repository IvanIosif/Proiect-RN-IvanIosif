import pandas as pd
import numpy as np
import os
import json
import joblib
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix

# Detectăm unde se află fișierul curent (ex: src/neural_network/train.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

PATH_BASE = os.path.abspath(os.path.join(current_dir, "../../"))

PATH_DATA = os.path.join(PATH_BASE, "data")
PATH_MODELS = os.path.join(PATH_BASE, "models")
PATH_RESULTS = os.path.join(PATH_BASE, "results")
PATH_CONFIG = os.path.join(PATH_BASE, "config")
PATH_DOCS = os.path.join(PATH_BASE, "docs")

for p in [PATH_MODELS, PATH_RESULTS, PATH_CONFIG, PATH_DOCS]:
    os.makedirs(p, exist_ok=True)

# --- 2. FUNCȚII ÎNCĂRCARE ȘI AUGMENTARE ---
def apply_augmentation(X):
    noise = np.random.normal(0, 0.02, X.shape)
    X_augmented = X + noise
    return np.clip(X_augmented, 0, 1)

def load_split(split_name, augment=False):
    # Căile către CSV-uri sunt acum relative la PATH_BASE
    p_path = os.path.join(PATH_DATA, split_name, "pneumonie", f"pneumonie_{split_name}.csv")
    t_path = os.path.join(PATH_DATA, split_name, "tuberculoza", f"tuberculoza_{split_name}.csv")
    
    if not os.path.exists(p_path) or not os.path.exists(t_path):
        raise FileNotFoundError(f"❌ Lipsesc datele în folderul: {os.path.relpath(p_path, PATH_BASE)}")

    df_p = pd.read_csv(p_path)
    df_t = pd.read_csv(t_path)
    df = pd.concat([df_p, df_t], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    X = df.drop('Diagnosis', axis=1).values
    y = df['Diagnosis'].values

    if augment:
        X_aug = apply_augmentation(X)
        X = np.concatenate([X, X_aug])
        y = np.concatenate([y, y])
    
    return X, y

# Încărcare date
print(f"📂 Încărcare date din: {PATH_DATA}")
X_train, y_train = load_split("train", augment=True)
X_val, y_val     = load_split("validation")
X_test, y_test   = load_split("test")

# --- 3. CONFIGURARE HIPERPARAMETRI ---
params = {
    "architecture": "Robust Perceptron Nivel 2",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50, 
    "optimizer": "Adam",
    "loss_function": "binary_crossentropy"
}

# --- 4. CONSTRUCȚIE MODEL ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(20,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
    loss=params["loss_function"],
    metrics=['accuracy']
)

# --- 5. CALLBACKS (OBLIGATORIU NIVEL 2) ---
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6,
    verbose=1
)

# --- 6. ANTRENARE ---
print("🚀 Pornire antrenare Nivel 2...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# --- 7. SALVARE REZULTATE VIZUALE ---
# Grafic Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss Antrenare', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Loss Validare (Val_Loss)', color='red', linestyle='--', linewidth=2)
plt.title('Evoluția Funcției de Loss - Nivel 2', fontsize=14)
plt.xlabel('Epoci', fontsize=12)
plt.ylabel('Valoare Loss', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
path_loss = os.path.join(PATH_DOCS, "loss_curve.png")
plt.savefig(path_loss)
plt.close()

# Predicții
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pneumonie', 'Tuberculoza'], 
            yticklabels=['Pneumonie', 'Tuberculoza'])
plt.title('Matricea de Confuzie - Nivel 2')
plt.xlabel('Predicție Model')
plt.ylabel('Realitate (Etichetă)')
path_cm = os.path.join(PATH_DOCS, "confusion_matrix.png")
plt.savefig(path_cm)
plt.close()

# --- 8. EXPORT ARTEFACTE ---
# Salvare Scaler (Meta-data)
joblib.dump({"status": "already_normalized", "features_count": 20}, os.path.join(PATH_CONFIG, "scaler.skl"))

# Salvare Model (.keras)
model.save(os.path.join(PATH_MODELS, "trained_model.keras"))

# Salvare Metrici și Hiperparametri
metrics = {
    "accuracy": round(float(np.mean(y_pred.flatten() == y_test)), 4),
    "f1_macro": round(float(f1_score(y_test, y_pred, average='macro')), 4),
    "epochs_until_stop": len(history.history['loss'])
}

with open(os.path.join(PATH_RESULTS, "test_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

with open(os.path.join(PATH_RESULTS, "hyperparameters.yaml"), 'w') as f:
    yaml.dump(params, f)

print(f"\n✅ Finalizat cu succes!")
print(f"📊 Rezultatele au fost salvate relativ la: {os.path.relpath(PATH_BASE, os.getcwd())}")
