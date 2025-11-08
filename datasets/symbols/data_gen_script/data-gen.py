import pandas as pd

D = 700
students = 11

data = []
for x in range(41, 101):
    X = students * x
    G = D + X
    G_eff = round(0.8 * G)  # 80% der Gesamtdatenpunkte
    teilbar = [str(n) for n in [2, 4, 8, 16, 32] if G_eff % n == 0]
    
    data.append({
        "x": x,
        "X = 11*x": X,
        "G = 700 + X": G,
        "G_eff = 0.8*G (gerundet)": G_eff,
        "G_eff mod 2": G_eff % 2,
        "G_eff mod 4": G_eff % 4,
        "G_eff mod 8": G_eff % 8,
        "G_eff mod 16": G_eff % 16,
        "G_eff mod 32": G_eff % 32,
        "Teilbar durch": ", ".join(teilbar) if teilbar else "-"
    })

df = pd.DataFrame(data)

# Tabelle anzeigen
print(df.to_string(index=False))
