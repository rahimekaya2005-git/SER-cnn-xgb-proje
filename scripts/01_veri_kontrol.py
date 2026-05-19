import os
import pandas as pd
import matplotlib.pyplot as plt

# proje yolu
BASE_DIR = "proje/yolu"

# dataset klasoru
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# ciktilar icin outputs klasoru
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# desteklenen ses uzantilari
audio_extensions = (".wav", ".mp3", ".flac", ".ogg")

data = []

print("Veri seti kontrol ediliyor...\n")

# dataset klasorundeki her duyguyu gezme
for emotion in os.listdir(DATASET_DIR):
    emotion_path = os.path.join(DATASET_DIR, emotion)
    if os.path.isdir(emotion_path):
        files = [
            f for f in os.listdir(emotion_path)
            if f.lower().endswith(audio_extensions)
        ]

        print(f"{emotion} sınıfı: {len(files)} ses dosyası")

        for file in files:
            file_path = os.path.join(emotion_path, file)
            data.append({
                "file": file_path,
                "emotion": emotion
            })

# DataFrame olusturma
df = pd.DataFrame(data)

print("\nToplam ses dosyası:", len(df))
print("\nSınıf dağılımı:")
print(df["emotion"].value_counts())

# CSV olarak kaydet
csv_path = os.path.join(OUTPUT_DIR, "01_dataset_kontrol.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"\nDataset kontrol dosyası kaydedildi: {csv_path}")

# sinif dagilimi grafigi
plt.figure(figsize=(8, 5))
df["emotion"].value_counts().plot(kind="bar")

plt.title("Duygu Sınıflarına Göre Ses Dosyası Dağılımı")
plt.xlabel("Duygu Sınıfı")
plt.ylabel("Dosya Sayısı")
plt.xticks(rotation=0)
plt.tight_layout()

grafik_path = os.path.join(OUTPUT_DIR, "01_class_distribution.png")
plt.savefig(grafik_path, dpi=300)
plt.show()

print(f"Sınıf dağılımı grafiği kaydedildi: {grafik_path}")