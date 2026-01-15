import streamlit as st
import pickle
import math
import os
import pandas as pd

# 1. Definiția Clasei - ESTE OBLIGATORIE pentru ca pickle să poată încărca modelul
class NeuralNetworkAbsoluteZero:
    def __init__(self, input_size=20, hidden_size=6, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W1, self.b1, self.W2, self.b2 = [], [], [], []

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-max(min(x, 50), -50)))

    def forward(self, x):
        # Stratul Ascuns (Hidden Layer)
        h = [self.sigmoid(sum(x[i] * self.W1[i][j] for i in range(self.input_size)) + self.b1[j]) 
             for j in range(self.hidden_size)]
        # Stratul de Ieșire (Output Layer)
        o = self.sigmoid(sum(h[j] * self.W2[j][0] for j in range(self.hidden_size)) + self.b2[0])
        return h, o

# 2. Funcție de încărcare a modelului salvat anterior
@st.cache_resource
def load_nn_model():
    # Calea trebuie să fie cea unde ai salvat modelul la antrenare
    model_path = r"D:\Facultate\RN\models\trained_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

# --- CONFIGURARE INTERFAȚĂ ---
st.set_page_config(page_title="Diagnostic SIA", page_icon="🧪", layout="centered")

st.title("🧪 Sistem Expert de Diagnostic")
st.write("Completează chestionarul de mai jos pentru a primi un diagnostic bazat pe rețeaua neuronală antrenată.")

# Încercăm să încărcăm modelul
nn = load_nn_model()

if nn is None:
    st.error("⚠️ Modelul `trained_model.pkl` nu a fost găsit în `D:\\Facultate\\RN\\models`. Te rugăm să rulezi scriptul de antrenare mai întâi!")
    st.stop()

# 3. Chestionarul (Cele 20 de întrebări)
questions = [
    {"id": "Q1", "text": "Cât de ridicată este febra?", "opts": ["Normală (36–37°C)", "Ușoară", "Moderată", "Ridicată", "Foarte ridicată"]},
    {"id": "Q2", "text": "Capacitate activități normale", "opts": ["Foarte puțin", "Puțin", "Moderat", "Mult", "Foarte mult"]},
    {"id": "Q3", "text": "Dificultate respirație", "opts": ["Deloc", "Foarte puțin", "Moderat", "Semnificativ", "Foarte greu"]},
    {"id": "Q4", "text": "Durată episod tuse", "opts": ["< 10s", "10-30s", "30s-2m", "2-4m", "> 4m"]},
    {"id": "Q5", "text": "Frecvență tuse", "opts": ["Foarte rar", "Ocazional", "Moderat", "Frecvent", "Foarte frecvent"]},
    {"id": "Q6", "text": "Durere în piept", "opts": ["Deloc", "Ușor", "Moderat", "Intens", "Foarte intens"]},
    {"id": "Q7", "text": "Tuse productivă", "opts": ["Deloc", "Foarte puțin", "Moderată", "Multă", "Foarte multă"]},
    {"id": "Q8", "text": "Frisoane", "opts": ["Deloc", "Ușor", "Moderat", "Puternic", "Foarte puternic"]},
    {"id": "Q9", "text": "Dureri de cap", "opts": ["Niciodată", "Rareori", "Uneori", "Des", "Foarte des"]},
    {"id": "Q10", "text": "Durere musculară", "opts": ["Deloc", "Ușoară", "Moderată", "Puternică", "Foarte puternică"]},
    {"id": "Q11", "text": "Transpirații nocturne", "opts": ["Niciodată", "Foarte rar", "Ocazional", "Frecvent", "Permanent"]},
    {"id": "Q12", "text": "Respirație întins pe spate", "opts": ["Deloc", "Foarte puțin", "Moderat", "Mult", "Foarte Mult"]},
    {"id": "Q13", "text": "Greață/Dureri abdominale", "opts": ["Niciodată", "Rareori", "Ocazional", "Frecvent", "Foarte frecvent"]},
    {"id": "Q14", "text": "Pierdere gust/miros", "opts": ["Deloc", "Foarte ușoară", "Moderată", "Pronunțată", "Foarte pronunțată"]},
    {"id": "Q15", "text": "Kg pierdute (3 luni)", "opts": ["0 kg", "1–2 kg", "3–5 kg", "6–10 kg", ">10 kg"]},
    {"id": "Q16", "text": "Tuse cu sânge", "opts": ["Niciunul", "Foarte puține", "Puține", "Multe", "Foarte multe"]},
    {"id": "Q17", "text": "Efort respirație", "opts": ["Deloc", "Foarte puțin", "Moderat", "Mult", "Foarte mult"]},
    {"id": "Q18", "text": "Ganglioni inflamați", "opts": ["Niciodată", "Foarte rar", "Ocazional", "Frecvent", "Permanent"]},
    {"id": "Q19", "text": "Reducere poftă mâncare", "opts": ["Deloc", "Foarte puțin", "Moderată", "Foarte mult", "Nu mai mănânc"]},
    {"id": "Q20", "text": "Febră intermitentă", "opts": ["Niciodată", "Foarte rar", "Ocazional", "Des", "Foarte Des"]}
]

# Crearea formularului
with st.form("diagnostic_form"):
    st.subheader("Simptome pacient")
    user_inputs_norm = []
    
    # Afișare pe două coloane pentru a economisi spațiu
    col1, col2 = st.columns(2)
    
    for i, q in enumerate(questions):
        with (col1 if i < 10 else col2):
            choice = st.selectbox(f"{q['id']}. {q['text']}", q['opts'])
            
            # NORMALIZARE: Transformăm alegerea (1-5) în (0.0 - 1.0)
            val_numeric = q['opts'].index(choice) + 1
            val_norm = (val_numeric - 1) / 4.0
            user_inputs_norm.append(val_norm)
            
    submit_button = st.form_submit_button("Lansează Diagnostic")

# 4. Procesarea Rezultatului
if submit_button:
    # Trecem datele normalizate prin rețea
    _, probabilitate = nn.forward(user_inputs_norm)
    
    st.divider()
    st.subheader("Rezultat Analiză Model:")
    
    if probabilitate >= 0.5:
        st.error(f"### Diagnostic Sugerat: **TUBERCULOZĂ**")
        st.write(f"Scor de încredere (Tuberculoza): **{probabilitate*100:.2f}%**")
    else:
        st.success(f"### Diagnostic Sugerat: **PNEUMONIE**")
        st.write(f"Scor de încredere (Pneumonia): **{(1 - probabilitate)*100:.2f}%**")
    
    st.info("Sistemul a normalizat răspunsurile tale în format numeric $[0, 1]$ înainte de calcul.")