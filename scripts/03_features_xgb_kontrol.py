import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ana proje yolu
BASE_DIR = "proje/yolu"

features_path = os.path.join(BASE_DIR, "features", "02_features_xgb.csv")
clean_features_path = os.path.join(BASE_DIR, "features", "03_features_xgb_clean.csv")

outputs_dir = os.path.join(BASE_DIR, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

report_path = os.path.join(outputs_dir, "03_features_kontrol_raporu.txt")
class_graph_path = os.path.join(outputs_dir, "03_class_distribution_features.png")

# 02 de kaydedilen features_xgb.csv okuma
df = pd.read_csv(features_path)

print("Özellik dosyası yüklendi.")
print(f"Satır sayısı  : {df.shape[0]}")
print(f"Sütun sayısı  : {df.shape[1]}")

# temel kontroller
print("\nİlk 5 satır:")
print(df.head())

print("\nSütunlar:")
print(df.columns.tolist())

print("\nSınıf dağılımı:")
print(df["emotion"].value_counts())

# eksik deger var mi kontrolu
missing_count = df.isnull().sum().sum()

# aynı kayittan birden fazla var mi kontrolu (tekrar kayit)
duplicate_count = df.duplicated().sum()

# file sutununa gore tekrar kontrolu
duplicate_file_count = df["file"].duplicated().sum()

# sayisal sutun kontrolu
feature_columns = df.drop(columns=["file", "emotion"]).columns

non_numeric_columns = []

for col in feature_columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        non_numeric_columns.append(col)

print("\nSayısal olmayan özellik sütunları:")
print(non_numeric_columns)

# infinite deger var mi kontrolu
numeric_df = df[feature_columns]

inf_count = np.isinf(numeric_df.to_numpy()).sum()

print("\nEksik değer sayısı:", missing_count)
print("Sonsuz değer sayısı:", inf_count)
print("Tekrarlı satır sayısı:", duplicate_count)
print("Tekrarlı dosya sayısı:", duplicate_file_count)

# temizleme
df_clean = df.copy()

# sonsuz degerleri NaN (not a number) yap
df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

# eksik deger varsa sutun ortalamasiyla doldur
for col in feature_columns:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

# ayni dosyadan birden fazla varsa ilkini tut
df_clean.drop_duplicates(subset=["file"], keep="first", inplace=True)

# temizlenmis dosyayi kaydetme
df_clean.to_csv(clean_features_path, index=False, encoding="utf-8-sig")

print("\n✅ Temizlenmiş özellik dosyası kaydedildi:")
print(clean_features_path)

print(f"\nTemiz veri satır sayısı : {df_clean.shape[0]}")
print(f"Temiz veri sütun sayısı : {df_clean.shape[1]}")

# sinif dagilim grafigi
plt.figure(figsize=(8, 5))
df_clean["emotion"].value_counts().plot(kind="bar")

plt.title("Özellik Dosyasındaki Duygu Sınıfı Dağılımı")
plt.xlabel("Duygu Sınıfı")
plt.ylabel("Kayıt Sayısı")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(class_graph_path, dpi=300)
plt.show()

print("\n✅ Sınıf dağılımı grafiği kaydedildi:")
print(class_graph_path)

# rapor dosyasi olussturma
with open(report_path, "w", encoding="utf-8") as f:
    f.write("03 - ÖZELLİK DOSYASI KONTROL RAPORU\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Orijinal satır sayısı  : {df.shape[0]}\n")
    f.write(f"Orijinal sütun sayısı  : {df.shape[1]}\n")
    f.write(f"Temiz satır sayısı     : {df_clean.shape[0]}\n")
    f.write(f"Temiz sütun sayısı     : {df_clean.shape[1]}\n\n")

    f.write(f"Eksik değer sayısı     : {missing_count}\n")
    f.write(f"Sonsuz değer sayısı    : {inf_count}\n")
    f.write(f"Tekrarlı satır sayısı  : {duplicate_count}\n")
    f.write(f"Tekrarlı dosya sayısı  : {duplicate_file_count}\n\n")

    f.write("Sınıf dağılımı:\n")
    f.write(str(df_clean["emotion"].value_counts()))
    f.write("\n\n")

    f.write("Sayısal olmayan özellik sütunları:\n")
    if non_numeric_columns:
        f.write(str(non_numeric_columns))
    else:
        f.write("Yok")

print("\n✅ Kontrol raporu kaydedildi:")
print(report_path)