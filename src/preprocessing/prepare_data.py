import os        # <--- ACEASTA ESTE LINIA CARE LIPSEȘTE
import pandas as pd

def process_data():
    raw_base = r"D:\Facultate\RN\data\raw"
    proc_base = r"D:\Facultate\RN\data\processed"
    
    for boala in ["pneumonie", "tuberculoza"]:
        file_path = os.path.join(raw_base, boala, "cases.csv")
        if not os.path.exists(file_path): 
            print(f"⚠️ Fișierul nu a fost găsit la: {file_path}")
            continue
        
        df = pd.read_csv(file_path)
        features = [f"Q{i}" for i in range(1, 21)]
        
        # Normalizare 1-5 -> 0-1
        df[features] = (df[features] - 1) / 4
        
        save_path = os.path.join(proc_base, boala)
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(os.path.join(save_path, "processed.csv"), index=False)
    
    print("✅ Etapa 2: Date normalizate (0.0 - 1.0).")

if __name__ == "__main__":
    process_data()