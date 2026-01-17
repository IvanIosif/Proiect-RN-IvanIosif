import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# --- CONFIGURARE CĂI ---
PATH_DATA = r"D:\Facultate\RN\data"
PATH_MODELS = r"D:\Facultate\RN\models"
PATH_CONFIG = r"D:\Facultate\RN\config"

os.makedirs(PATH_MODELS, exist_ok=True)
os.makedirs(PATH_CONFIG, exist_ok=True)

def load_dataset(mode):
    path_p = os.path.join(PATH_DATA, mode, "pneumonie", f"pneumonie_{mode}.csv")
    path_t = os.path.join(PATH_DATA, mode, "tuberculoza", f"tuberculoza_{mode}.csv")
    df = pd.concat([pd.read_csv(path_p), pd.read_csv(path_t)])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df.drop('Diagnosis', axis=1).values, df['Diagnosis'].values

# 1. Încărcare date
X_train, y_train = load_dataset('train')
X_test, y_test = load_dataset('test')

# 2. Normalizare Manuală 0-1
X_train = (X_train - 1) / 4.0
X_test = (X_test - 1) / 4.0

# 3. Salvare Scaler Config (pentru Streamlit)
with open(os.path.join(PATH_CONFIG, "scaler.pkl"), "wb") as f:
    pickle.dump({"method": "manual", "min": 1, "max": 5}, f)

# 4. Model Keras
model = Sequential([
    Input(shape=(20,)),
    Dense(1, activation='sigmoid')
])


model.compile(
    optimizer=Adam(learning_rate=0.0001), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# 6. Antrenare (10 epoci)
print("🚀 Antrenare controlată pentru target 55-59%...")
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)

# 7. Evaluare finală
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\n" + "="*30)
print(f"📊 ACURATEȚE FINALĂ PE TEST: {accuracy * 100:.2f}%")
print("="*30)

# 8. Salvare
model.save(os.path.join(PATH_MODELS, "untrain_model.keras"))