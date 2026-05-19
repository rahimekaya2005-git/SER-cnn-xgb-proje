import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical

# ana proje yolu
BASE_DIR = Path(r"proje/yolu")

SPEKTROGRAM_DIR = BASE_DIR / "features" / "05_mel_spektrogramlar"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
# xgb de ayrilan test train dosya yolu
SPLIT_PATH = OUTPUTS_DIR / "04_common_train_test_split.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

TEST_SIZE = 0.2
VAL_SIZE = 0.1

SINIFLAR = ["Angry", "Calm", "Happy", "Sad"]

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)

# veri yukleme : spektrogramlari okur her gorseli 224x224 numpy dizisine çevirir
def veri_yukle():
    gorseller = []
    etiketler = []
    dosya_yollari = []
    splits = []

    split_df = pd.read_csv(SPLIT_PATH)
    # stem sütunu yoksa file sütunundan oluştur
    if "stem" not in split_df.columns:
        split_df["stem"] = split_df["file"].apply(lambda x: Path(x).stem)
    split_map = dict(zip(split_df["stem"], split_df["split"]))

    for sinif in SINIFLAR:
        sinif_klasor = SPEKTROGRAM_DIR / sinif

        if not sinif_klasor.exists():
            print(f"UYARI: {sinif_klasor} klasörü bulunamadı.")
            continue

        png_dosyalari = list(sinif_klasor.rglob("*.png"))

        print(f"{sinif}: {len(png_dosyalari)} görsel bulundu")

        for dosya in png_dosyalari:
            try:
                dosya_stem = dosya.stem

                if dosya_stem not in split_map:
                    print(f"  UYARI: {dosya.name} split dosyasında bulunamadı, atlandı.")
                    continue

                img = Image.open(dosya).convert("RGB")
                img = img.resize(IMG_SIZE)

                gorseller.append(np.array(img))
                etiketler.append(sinif)
                dosya_yollari.append(str(dosya))
                splits.append(split_map[dosya_stem])

            except Exception as hata:
                print(f"  HATA: {dosya.name} -> {hata}")

    X = np.array(gorseller, dtype="float32") / 255.0
    y = np.array(etiketler)
    files = np.array(dosya_yollari)
    splits = np.array(splits)

    return X, y, files, splits

# veri hazirlama
def veri_hazirla():
    print("Veriler yükleniyor...\n")

    # veri_yukle artık 4 değer döndürüyor:
    # X, etiketler, dosya yolları, train/test split bilgisi
    X, y_ham, files, split_labels = veri_yukle()

    print(f"\nToplam {len(X)} görsel yüklendi")
    print(f"Görsel veri boyutu: {X.shape}")

    if len(X) == 0:
        raise ValueError("Hiç görsel yüklenmedi. Spektrogram klasörünü kontrol et.")

    # label encoder
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_ham)

    label_encoder_path = MODELS_DIR / "06_label_encoder_cnn.pkl"
    joblib.dump(encoder, label_encoder_path)

    print("\nSınıf eşleştirmeleri:")
    for i, sinif in enumerate(encoder.classes_):
        print(f"  {sinif} -> {i}")

    print("\nSınıf dağılımı:")
    for sinif, sayi in zip(*np.unique(y_ham, return_counts=True)):
        print(f"  {sinif:<10} → {sayi} görsel")

    # XGBoost ile aynı train-test ayrımını uygulama
    train_val_mask = split_labels == "train"
    test_mask = split_labels == "test"

    X_train_val = X[train_val_mask]
    X_test = X[test_mask]

    y_train_val = y_encoded[train_val_mask]
    y_test_encoded = y_encoded[test_mask]

    files_train_val = files[train_val_mask]
    files_test = files[test_mask]

    print("\nOrtak split kontrolü:")
    print(f"Train + Validation : {len(X_train_val)} görsel")
    print(f"Test               : {len(X_test)} görsel")

    if len(X_train_val) == 0 or len(X_test) == 0:
        raise ValueError(
            "Train veya test seti boş oluştu. "
            "04_common_train_test_split.csv dosyasındaki split değerlerini kontrol et."
        )

    # train seti içinden validation ayırma
    X_train, X_val, y_train_encoded, y_val_encoded, files_train, files_val = train_test_split(
        X_train_val,
        y_train_val,
        files_train_val,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_val
    )

    y_train = to_categorical(y_train_encoded, num_classes=len(encoder.classes_))
    y_val = to_categorical(y_val_encoded, num_classes=len(encoder.classes_))
    y_test = to_categorical(y_test_encoded, num_classes=len(encoder.classes_))

    print(f"\nEğitim seti    : {len(X_train)} görsel")
    print(f"Validation seti: {len(X_val)} görsel")
    print(f"Test seti      : {len(X_test)} görsel")

    print("\nTest sınıf dağılımı:")
    for sinif, sayi in zip(*np.unique(y_test_encoded, return_counts=True)):
        print(f"  {encoder.classes_[sinif]:<10} → {sayi} görsel")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        y_train_encoded, y_val_encoded, y_test_encoded,
        files_test,
        encoder
    )

# cnn modeli
def model_olustur(sinif_sayisi):
    model = models.Sequential([
        Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(sinif_sayisi, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# cnn model egitimi
def egit(model, X_train, X_val, y_train, y_val, y_train_encoded):
    print("\nModel eğitiliyor...")

    # sin,flar dengeli olmadigi icin class_weight kullanimi
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_encoded),
        y=y_train_encoded
    )

    class_weights = {
        i: weight for i, weight in enumerate(class_weights_array)
    }

    print("\nClass weight değerleri:")
    for sinif_no, weight in class_weights.items():
        print(f"  Sınıf {sinif_no}: {weight:.4f}")

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint_path = MODELS_DIR / "06_cnn_best_model.keras"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    gecmis = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, lr_scheduler, checkpoint],
        class_weight=class_weights
    )

    return gecmis

# egitim grafigi
def egitim_grafigi_kaydet(gecmis):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(gecmis.history["accuracy"], label="Eğitim")
    ax1.plot(gecmis.history["val_accuracy"], label="Validation")
    ax1.set_title("CNN Model Doğruluğu")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(gecmis.history["loss"], label="Eğitim")
    ax2.plot(gecmis.history["val_loss"], label="Validation")
    ax2.set_title("CNN Model Kaybı")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()

    grafik_path = OUTPUTS_DIR / "06_cnn_egitim_grafigi.png"
    plt.savefig(grafik_path, dpi=300)
    plt.close()

    print(f"\nEğitim grafiği kaydedildi: {grafik_path}")


# confusion matrix
def confusion_matrix_kaydet(y_true, y_pred_classes, encoder):
    cm = confusion_matrix(y_true, y_pred_classes)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    plt.title("CNN - Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    cm_path = OUTPUTS_DIR / "06_cnn_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"Confusion matrix kaydedildi: {cm_path}")

# sonuclari kaydetme
def sonuclari_kaydet(model, X_test, y_test, y_test_encoded, files_test, encoder):
    print("\nTest verisi üzerinde değerlendirme yapılıyor...")

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    y_pred_proba = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test_encoded, y_pred_classes)
    precision_macro = precision_score(y_test_encoded, y_pred_classes, average="macro", zero_division=0)
    recall_macro = recall_score(y_test_encoded, y_pred_classes, average="macro", zero_division=0)
    f1_macro = f1_score(y_test_encoded, y_pred_classes, average="macro", zero_division=0)

    precision_weighted = precision_score(y_test_encoded, y_pred_classes, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test_encoded, y_pred_classes, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_test_encoded, y_pred_classes, average="weighted", zero_division=0)

    print("\n--- CNN TEST SONUÇLARI ---")
    print(f"Accuracy        : %{accuracy * 100:.2f}")
    print(f"Precision Macro : %{precision_macro * 100:.2f}")
    print(f"Recall Macro    : %{recall_macro * 100:.2f}")
    print(f"F1-Macro        : %{f1_macro * 100:.2f}")
    print(f"F1-Weighted     : %{f1_weighted * 100:.2f}")
    print(f"Test Loss       : {test_loss:.4f}")

    print("\n--- SINIFLANDIRMA RAPORU ---\n")

    report_text = classification_report(
        y_test_encoded,
        y_pred_classes,
        target_names=encoder.classes_,
        zero_division=0
    )

    print(report_text)

    # Classification report CSV
    report_dict = classification_report(
        y_test_encoded,
        y_pred_classes,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_path = OUTPUTS_DIR / "06_cnn_classification_report.csv"
    report_df.to_csv(report_path, encoding="utf-8-sig")

    # metrik ozeti
    metrics_summary = pd.DataFrame([{
        "model": "CNN",
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "test_loss": test_loss
    }])

    metrics_path = OUTPUTS_DIR / "06_cnn_metrics_summary.csv"
    metrics_summary.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    # test tahminleri
    predictions_df = pd.DataFrame({
        "file": files_test,
        "true_label": encoder.inverse_transform(y_test_encoded),
        "predicted_label": encoder.inverse_transform(y_pred_classes),
        "confidence": np.max(y_pred_proba, axis=1)
    })

    for i, class_name in enumerate(encoder.classes_):
        predictions_df[f"prob_{class_name}"] = y_pred_proba[:, i]

    predictions_path = OUTPUTS_DIR / "06_cnn_test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    confusion_matrix_kaydet(y_test_encoded, y_pred_classes, encoder)

    print("\nCNN çıktı dosyaları kaydedildi:")
    print(f"Classification report : {report_path}")
    print(f"Metrik özeti          : {metrics_path}")
    print(f"Test tahminleri       : {predictions_path}")

# ana fonksiyon
def main():
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        y_train_encoded, y_val_encoded, y_test_encoded,
        files_test,
        encoder
    ) = veri_hazirla()

    sinif_sayisi = len(encoder.classes_)

    model = model_olustur(sinif_sayisi)
    model.summary()

    gecmis = egit(
        model,
        X_train,
        X_val,
        y_train,
        y_val,
        y_train_encoded
    )

    sonuclari_kaydet(
        model,
        X_test,
        y_test,
        y_test_encoded,
        files_test,
        encoder
    )

    egitim_grafigi_kaydet(gecmis)

    final_model_path = MODELS_DIR / "06_cnn_model.keras"
    model.save(final_model_path)

    print(f"\nModel kaydedildi: {final_model_path}")


if __name__ == "__main__":
    main()