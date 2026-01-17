import pandas as pd
import os
from sklearn.model_selection import train_test_split

def final_split_and_distribute():
    # Căile tale
    proc_base = r"D:\Facultate\RN\data\processed"
    final_base = r"D:\Facultate\RN\data"
    
    # Încărcare date procesate (normalizate anterior 0-1)
    df_p = pd.read_csv(os.path.join(proc_base, "pneumonie", "processed.csv"))
    df_t = pd.read_csv(os.path.join(proc_base, "tuberculoza", "processed.csv"))
    
    # Unire pentru un shuffle global (esențial ca modelul să nu vadă pattern-uri)
    df_full = pd.concat([df_p, df_t]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Split 70% Train, 30% Rest
    train_df, temp_df = train_test_split(df_full, test_size=0.30, stratify=df_full['Diagnosis'], random_state=42)
    # Split 30% rest în 15% Val și 15% Test
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['Diagnosis'], random_state=42)

    sets = {
        'train': train_df,
        'validation': val_df,
        'test': test_df
    }

    for mode, data in sets.items():
        for label, name in [(0, "pneumonie"), (1, "tuberculoza")]:
            subset = data[data['Diagnosis'] == label]
            
            # Creează folderul: D:\Facultate\RN\data\{mode}\{name}
            folder_path = os.path.join(final_base, mode, name)
            os.makedirs(folder_path, exist_ok=True)
            
            # SALVARE: Numele fișierului trebuie să fie exact ca în load_dataset
            # Exemplu: pneumonie_train.csv
            file_name = f"{name}_{mode}.csv"
            subset.to_csv(os.path.join(folder_path, file_name), index=False)
            
    print("✅ Fișierele au fost distribuite corect!")

if __name__ == "__main__":
    final_split_and_distribute()