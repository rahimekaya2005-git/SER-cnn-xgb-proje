import os
import numpy as np
import pandas as pd
import opensmile
import librosa

# ana proje yolu
BASE_DIR = "proje/yolu"

dataset_path = os.path.join(BASE_DIR, "dataset")
features_dir = os.path.join(BASE_DIR, "features")

os.makedirs(features_dir, exist_ok=True)

output_csv_path = os.path.join(features_dir, "02_features_xgb.csv")

# opensmile nesnesi olusturma
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

# mfcc ozellik cikarma fonksiyonu
def extract_mfcc_features(file_path, n_mfcc=20):
    mfcc_features = {}
    # ses dosyasını yükleme
    # sr=16000: tum sesleri aynı ornekleme frekansina getirir
    # mono=True: stereo sesleri tek kanala cevirir
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    # MFCC çıkarımı
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )

    # MFCC delta ve delta-delta özellikleri
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # her MFCC katsayisi icin ozet istatistikler
    for i in range(n_mfcc):
        mfcc_features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        mfcc_features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])
        mfcc_features[f"mfcc_{i+1}_min"] = np.min(mfcc[i])
        mfcc_features[f"mfcc_{i+1}_max"] = np.max(mfcc[i])

        mfcc_features[f"mfcc_delta_{i+1}_mean"] = np.mean(mfcc_delta[i])
        mfcc_features[f"mfcc_delta_{i+1}_std"] = np.std(mfcc_delta[i])

        mfcc_features[f"mfcc_delta2_{i+1}_mean"] = np.mean(mfcc_delta2[i])
        mfcc_features[f"mfcc_delta2_{i+1}_std"] = np.std(mfcc_delta2[i])

    return mfcc_features


# tum ses dosyalarindan ozellik cikarma
all_features = []

print("Özellik çıkarımı başlatıldı...\n")

# dataset klasorundeki duygulari gezme
for emotion_folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, emotion_folder)

    # klasor degilse atla
    if not os.path.isdir(folder_path):
        continue

    print(f"\n{emotion_folder} sınıfı işleniyor...")

    # duygu klasorundeki ses dosyalarini gez
    for file in os.listdir(folder_path):
        if file.lower().endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            try:
                # opensmile ozellikleri
                opensmile_df = smile.process_file(file_path)

                # opensmile ozelliklerini tek satira cevirme
                opensmile_features = opensmile_df.reset_index(drop=True).iloc[0].to_dict()

                # mfcc ozellikleri
                mfcc_features = extract_mfcc_features(file_path, n_mfcc=20)

                # tum ozellikleri birleştirme
                combined_features = {}

                combined_features.update(opensmile_features)
                combined_features.update(mfcc_features)

                # Ddsya ve duygu etiketi ekleme
                combined_features["file"] = file_path
                combined_features["emotion"] = emotion_folder

                all_features.append(combined_features)

                print(f"✔ {file} işlendi")
            except Exception as e:
                print(f"❌ {file} işlenemedi: {e}")


# csv dosyasi olarak kaydetme
if all_features:
    final_df = pd.DataFrame(all_features)

    # file ve emotion sutunlarını en basa alma
    first_columns = ["file", "emotion"]
    other_columns = [col for col in final_df.columns if col not in first_columns]
    final_df = final_df[first_columns + other_columns]

    # CSV kaydetme
    final_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print("\n✅ Tüm özellikler başarıyla çıkarıldı.")
    print(f"✅ CSV dosyası kaydedildi: {output_csv_path}")
    print(f"Toplam kayıt sayısı: {final_df.shape[0]}")
    print(f"Toplam sütun sayısı: {final_df.shape[1]}")

else:
    print("⚠ Hiçbir dosya işlenmedi.")