import pandas as pd
import numpy as np
import os
import random

def generate_massive_chaotic_dataset():
    # Seed variabil pentru a evita seturi identice la rulări diferite
    np.random.seed(random.randint(1, 10000)) 
    path_raw = r"D:\Facultate\RN\docs\raw"
    
    os.makedirs(os.path.join(path_raw, "Pneumonie"), exist_ok=True)
    os.makedirs(os.path.join(path_raw, "Tuberculoza"), exist_ok=True)

    def get_noisy_val(base_val, spread=2.6): 
        # Spread-ul de 2.6 menține un nivel ridicat de haos
        noise = np.random.normal(0, spread)
        return int(np.clip(round(base_val + noise), 1, 5))

    def create_samples(label, count):
        samples = []
        for _ in range(count):
            row = {}
            
            # --- BLOCURI DE CORELAȚIE LOGICĂ ---
            
            # Bloc 1: Infecție sistemică (Febra Q1 și Frisoane Q8)
            fever_base = 3.5 if label == 0 else 3.0
            row['Q1'] = get_noisy_val(fever_base, spread=2.0)
            row['Q8'] = get_noisy_val(row['Q1'], spread=1.1) # Frisoanele urmează febra

            # Bloc 2: Afectare respiratorie cronică (TBC)
            # Transpirații nocturne (Q11) și Febra intermitentă (Q20)
            tbc_chronic_base = 3.8 if label == 1 else 2.2
            row['Q11'] = get_noisy_val(tbc_chronic_base, spread=2.2)
            row['Q20'] = get_noisy_val(row['Q11'], spread=1.4)

            # Bloc 3: Stare generală (Slăbiciune Q2 vs Efort respirație Q17)
            row['Q17'] = get_noisy_val(3.5 if label == 1 else 3.0, spread=2.0)
            # Dacă depui efort mare să respiri, activitatea (Q2) scade
            row['Q2'] = get_noisy_val(6 - row['Q17'], spread=1.2)

            # Bloc 4: Tuse și Sânge (Specific TBC sever)
            row['Q16'] = get_noisy_val(3.3 if label == 1 else 1.5, spread=2.5)
            # Dacă avem sânge, durata tusei (Q4) e mare
            row['Q4'] = get_noisy_val(row['Q16'] if row['Q16'] > 2 else 2.5, spread=1.8)

            # --- COMPLETRE REST (Haos Controlat) ---
            for i in range(1, 21):
                q_key = f'Q{i}'
                if q_key in row: continue
                
                # Balansăm restul întrebărilor pe baza etichetei
                base = 3.1 if label == 0 else 2.7
                row[q_key] = get_noisy_val(base, spread=2.9) # Spread maxim pentru haos

            samples.append([row[f'Q{i}'] for i in range(1, 21)] + [label])
        return samples

    cols = [f"Q{i}" for i in range(1, 21)] + ["Diagnosis"]
    
    print(f"🚀 Începe generarea a 30.000 de instanțe...")
    
    pne_data = create_samples(0, 15000)
    print("✅ 15.000 Pneumonie generate.")
    
    tbc_data = create_samples(1, 15000)
    print("✅ 15.000 Tuberculoza generate.")
    
    # Salvare în fișiere separate conform structurii tale
    pd.DataFrame(pne_data, columns=cols).to_csv(os.path.join(path_raw, "Pneumonie", "cases.csv"), index=False)
    pd.DataFrame(tbc_data, columns=cols).to_csv(os.path.join(path_raw, "Tuberculoza", "cases.csv"), index=False)
    
    print(f"\n✨ Succes! Fișierele au fost salvate în:\n{path_raw}")

if __name__ == "__main__":
    generate_massive_chaotic_dataset()
