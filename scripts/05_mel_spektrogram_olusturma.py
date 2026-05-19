# kutuphaneler
import os
import numpy as np
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")  # ekran olmadan calistirmak icin

import matplotlib.pyplot as plt
from pathlib import Path

# klasor yollari
DATASET_DIR = Path(r"proje/yolu/dataset")  # veri setinin cekildigi yol
OUTPUT_DIR  = Path(r"proje/yolu/features/05_mel_spektrogramlar")  # spektrogramlarin kaydedilecegi yol

# mel spektrogram parametreleri
N_MELS     = 128
HOP_LENGTH = 512
N_FFT      = 2048
FMAX       = 8000  # konusma sesi icin 8khz yeterli
# cnn icin sabit ses ayarlari
TARGET_SR = 22050
TARGET_SEC = 1

# veri setindeki ses dosyalarini yukler, normalize eder
def yukle(dosya_yolu):
    # Tüm sesleri aynı örnekleme hızına getiriyoruz
    y, sr = librosa.load(dosya_yolu, sr=TARGET_SR, mono=True)

    if len(y) == 0:
        raise ValueError("ses dosyasi bos!")

    # Tüm sesleri aynı süreye sabitle
    hedef_uzunluk = TARGET_SR * TARGET_SEC

    if len(y) < hedef_uzunluk:
        # Ses kısa ise sonuna sıfır ekle
        y = np.pad(y, (0, hedef_uzunluk - len(y)))
    else:
        # Ses uzunsa kes
        y = y[:hedef_uzunluk]

    # Normalize et
    y = y / (np.max(np.abs(y)) + 1e-9)

    return y, sr

# ses sinyalinin mel-spektrogramini hesaplar, dB olcegine cevirir
def mel_spektrogram_hesapla(y, sr):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=FMAX
    )

    # power_to_db ile log olcegine cevir
    # top_db=80 ; dinamik araligi 80 dB ile sinirla, cok karanlik olmamasi icin
    S_dB = librosa.power_to_db(S, ref=np.max, top_db=80)
    return S_dB

# hesaplanan spektrogrami gorsellestirir, png olarak kaydeder
def kaydet(S_dB, sr, baslik, cikti_yolu):
    # 1. Sabit bir figür boyutu belirle (İnç cinsinden)
    # Formül: Piksel / DPI = İnç. Örn: 224 piksel / 100 dpi = 2.24 inç
    pixel_size = 224
    my_dpi = 100
    figsize = pixel_size / my_dpi

    fig = plt.figure(figsize=(figsize, figsize), dpi=my_dpi)
    ax = fig.add_axes([0, 0, 1, 1]) # Tüm alanı kapla (kenar boşluğu bırakma)
    ax.set_axis_off()

    # 2. Spektrogramı çiz
    librosa.display.specshow(
        S_dB,
        sr=sr,
        hop_length=HOP_LENGTH,
        fmax=FMAX,
        x_axis = None,
        y_axis = None,
        ax=ax
    )

    # 3. Klasör kontrolü
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)

    # 4. Kaydetme ayarları
    # bbox_inches="tight" kullanmıyoruz çünkü boyutu değiştirebilir!
    fig.savefig(cikti_yolu, dpi=my_dpi)
    plt.close(fig)

# veri setindeki tum ses dosyalarini gezerek yukaridaki fonsiyonlari cagirip sonucu gosterir
def main():
    if not DATASET_DIR.exists():
        print(f"HATA: klasor bulunamadi -> {DATASET_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    duygular = [d for d in DATASET_DIR.iterdir() if d.is_dir()]
    print(f"toplam {len(duygular)} duygu klasoru bulundu")

    basarili = 0
    basarisiz = 0

    for duygu_klasoru in duygular:
        duygu_adi = duygu_klasoru.name
        print(f"\n-- {duygu_adi} isleniyor --")

        dosyalar = [f for f in duygu_klasoru.iterdir()
                    if f.suffix.lower() in (".wav", ".mp3", ".flac")]

        for dosya in dosyalar:
            cikti = OUTPUT_DIR / duygu_adi / dosya.with_suffix(".png").name

            # zaten varsa tekrar isleme
            if cikti.exists():
                continue

            try:
                y, sr  = yukle(dosya)
                S_dB   = mel_spektrogram_hesapla(y, sr)
                baslik = f"{duygu_adi} - {dosya.name}"
                kaydet(S_dB, sr, baslik, cikti)

                print(f"  ok: {dosya.name}")
                basarili += 1

            except Exception as hata:
                print(f"  HATA: {dosya.name} -> {hata}")
                basarisiz += 1

    print(f"\nBitti! Basarili: {basarili}, Hata: {basarisiz}")


if __name__ == "__main__":
    main()