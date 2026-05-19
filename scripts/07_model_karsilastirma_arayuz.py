import os
import tempfile
from io import BytesIO
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import opensmile
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# proje yollari
BASE_DIR = Path(r"proje/yolu")
MODELS_DIR = BASE_DIR / "models"
FEATURES_DIR = BASE_DIR / "features"
OUTPUTS_DIR = BASE_DIR / "outputs"

XGB_PIPELINE_PATH = MODELS_DIR / "04_xgb_pipeline.pkl"
XGB_ENCODER_PATH = MODELS_DIR / "04_label_encoder_xgb.pkl"

CNN_MODEL_PATH = MODELS_DIR / "06_cnn_best_model.keras"
CNN_ENCODER_PATH = MODELS_DIR / "06_label_encoder_cnn.pkl"

XGB_FEATURE_TEMPLATE_PATH = FEATURES_DIR / "03_features_xgb_clean.csv"

XGB_METRICS_PATH = OUTPUTS_DIR / "04_xgb_metrics_summary.csv"
CNN_METRICS_PATH = OUTPUTS_DIR / "06_cnn_metrics_summary.csv"

XGB_CM_PATH = OUTPUTS_DIR / "04_xgb_confusion_matrix.png"
CNN_CM_PATH = OUTPUTS_DIR / "06_cnn_confusion_matrix.png"
CNN_TRAIN_GRAPH_PATH = OUTPUTS_DIR / "06_cnn_egitim_grafigi.png"
XGB_FEATURE_GRAPH_PATH = OUTPUTS_DIR / "04_xgb_feature_importance.png"

# ifadeler
EMOTION_EMOJIS = {
    "Angry": "😠",
    "Calm": "😌",
    "Happy": "😄",
    "Sad": "😢"
}

# grafiklerde metrik renkleri
METRIC_COLORS = {
    "Accuracy": "#7FB3D5",      # pastel mavi
    "Precision": "#82E0AA",    # pastel yeşil
    "Recall": "#F7DC6F",       # pastel sarı
    "F1-Macro": "#F5B7B1",     # pastel pembe
    "F1-Weighted": "#C39BD3"   # pastel mor
}

PROBABILITY_BAR_COLOR = "#85C1E9"

# xgb ozellik ayarlari
XGB_MFCC_SR = 16000
XGB_N_MFCC = 20

# cnn mel-spektrogram ayarlari
IMG_SIZE = (224, 224)
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
FMAX = 8000
TARGET_SR = 22050
TARGET_SEC = 1

# streamlit sayfa ayarlari
st.set_page_config(
    page_title="Sesten Duygu Tanıma | XGBoost - CNN",
    page_icon="🎧",
    layout="wide"
)
st.title("🎧 Sesten Duygu Tanıma Arayüzü")
st.caption("XGBoost ve CNN modelleri ile karşılaştırmalı duygu tahmini")

# model yukleme
@st.cache_resource
def load_xgb_pipeline():
    return joblib.load(XGB_PIPELINE_PATH)

@st.cache_resource
def load_xgb_encoder():
    return joblib.load(XGB_ENCODER_PATH)

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(CNN_MODEL_PATH)

@st.cache_resource
def load_cnn_encoder():
    return joblib.load(CNN_ENCODER_PATH)

@st.cache_resource
def load_opensmile_model():
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )

@st.cache_data
def load_xgb_feature_columns():
    df = pd.read_csv(XGB_FEATURE_TEMPLATE_PATH, nrows=1)
    return [col for col in df.columns if col not in ["file", "emotion"]]


# xgb ozellik cikarma
def extract_mfcc_features(file_path, n_mfcc=XGB_N_MFCC):
    mfcc_features = {}

    y, sr = librosa.load(file_path, sr=XGB_MFCC_SR, mono=True)

    if len(y) == 0:
        raise ValueError("Ses dosyası boş.")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

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


def extract_xgb_features(file_path):
    smile = load_opensmile_model()
    feature_columns = load_xgb_feature_columns()

    opensmile_df = smile.process_file(file_path)
    opensmile_features = opensmile_df.reset_index(drop=True).iloc[0].to_dict()

    mfcc_features = extract_mfcc_features(file_path)

    combined_features = {}
    combined_features.update(opensmile_features)
    combined_features.update(mfcc_features)

    feature_df = pd.DataFrame([combined_features])

    for col in feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0

    feature_df = feature_df[feature_columns]
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return feature_df


# cnn mel-spektrogram
def load_audio_for_cnn(file_path):
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

    if len(y) == 0:
        raise ValueError("Ses dosyası boş.")

    target_length = TARGET_SR * TARGET_SEC

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    y = y / (np.max(np.abs(y)) + 1e-9)
    return y, sr


def calculate_mel_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=FMAX
    )
    return librosa.power_to_db(S, ref=np.max, top_db=80)


def mel_spectrogram_to_image(S_dB, sr):
    pixel_size = 224
    my_dpi = 100
    figsize = pixel_size / my_dpi

    fig = plt.figure(figsize=(figsize, figsize), dpi=my_dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    librosa.display.specshow(
        S_dB,
        sr=sr,
        hop_length=HOP_LENGTH,
        fmax=FMAX,
        x_axis=None,
        y_axis=None,
        ax=ax
    )

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=my_dpi)
    plt.close(fig)
    buffer.seek(0)

    img = Image.open(buffer).convert("RGB")
    img = img.resize(IMG_SIZE)
    return img


def prepare_cnn_input(file_path):
    y, sr = load_audio_for_cnn(file_path)
    S_dB = calculate_mel_spectrogram(y, sr)
    img = mel_spectrogram_to_image(S_dB, sr)

    X_cnn = np.array(img, dtype="float32") / 255.0
    X_cnn = np.expand_dims(X_cnn, axis=0)

    return X_cnn, img


# tahmin
def predict_xgb(file_path):
    xgb_pipeline = load_xgb_pipeline()
    xgb_encoder = load_xgb_encoder()

    X_xgb = extract_xgb_features(file_path)

    pred_encoded = xgb_pipeline.predict(X_xgb)[0]
    pred_proba = xgb_pipeline.predict_proba(X_xgb)[0]
    pred_label = xgb_encoder.inverse_transform([pred_encoded])[0]

    return pred_label, pred_proba, xgb_encoder.classes_


def predict_cnn(file_path):
    cnn_model = load_cnn_model()
    cnn_encoder = load_cnn_encoder()

    X_cnn, spectrogram_img = prepare_cnn_input(file_path)

    pred_proba = cnn_model.predict(X_cnn, verbose=0)[0]
    pred_encoded = int(np.argmax(pred_proba))
    pred_label = cnn_encoder.inverse_transform([pred_encoded])[0]

    return pred_label, pred_proba, cnn_encoder.classes_, spectrogram_img


# grafik fonksiyonlari
def make_probability_df(classes, probabilities):
    df = pd.DataFrame({
        "Duygu": classes,
        "Olasılık": probabilities * 100
    }).sort_values("Olasılık", ascending=False)
    return df

def plot_probability_chart(prob_df, title):
    fig, ax = plt.subplots(figsize=(6, 4))

    labels = [
        f"{EMOTION_EMOJIS.get(row['Duygu'], '')} {row['Duygu']}"
        for _, row in prob_df.iterrows()
    ]

    values = prob_df["Olasılık"].values

    bars = ax.bar(labels, values, color=PROBABILITY_BAR_COLOR)

    ax.set_title(title)
    ax.set_ylabel("Olasılık (%)")
    ax.set_ylim(0, 100)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    return fig

def plot_metrics_for_model(model_name, metrics_dict):
    labels = ["Accuracy", "Precision", "Recall", "F1-Macro", "F1-Weighted"]

    values = [
        metrics_dict.get("accuracy", 0) * 100,
        metrics_dict.get("precision_macro", 0) * 100,
        metrics_dict.get("recall_macro", 0) * 100,
        metrics_dict.get("f1_macro", 0) * 100,
        metrics_dict.get("f1_weighted", 0) * 100
    ]

    colors = [METRIC_COLORS[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    bars = ax.bar(labels, values, color=colors)

    ax.set_title(f"{model_name} Metrik Grafiği")
    ax.set_ylabel("Değer (%)")
    ax.set_ylim(0, 100)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.xticks(rotation=20)
    plt.tight_layout()
    return fig

def load_metrics_data():
    xgb_metrics = None
    cnn_metrics = None

    if XGB_METRICS_PATH.exists():
        xgb_metrics = pd.read_csv(XGB_METRICS_PATH).iloc[0].to_dict()

    if CNN_METRICS_PATH.exists():
        cnn_metrics = pd.read_csv(CNN_METRICS_PATH).iloc[0].to_dict()

    return xgb_metrics, cnn_metrics


def emotion_text(label):
    return f"{EMOTION_EMOJIS.get(label, '🎵')} {label}"

#  arayuzun sekme yapisi
tab1, tab2, tab3 = st.tabs([
    "🎙️ Ses Tahmini",
    "📊 Model Metrikleri",
    "🖼️ Sonuç Görselleri"
])

# 1. sekme : tahmin
with tab1:
    st.header("Ses dosyası ile duygu tahmini")

    uploaded_file = st.file_uploader(
        "Bir ses dosyası yükle",
        type=["wav", "mp3", "flac", "ogg"]
    )

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(uploaded_file.getbuffer())
            temp_audio_path = temp_audio.name

        try:
            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.subheader("Yüklenen ses")
                st.audio(uploaded_file)

            with st.spinner("Spektrogram oluşturuluyor..."):
                _, spectrogram_img = prepare_cnn_input(temp_audio_path)

            with right_col:
                st.subheader("Mel-spektrogram")
                st.image(spectrogram_img, width=300)

            with st.spinner("Modeller tahmin yapıyor..."):
                xgb_label, xgb_proba, xgb_classes = predict_xgb(temp_audio_path)
                cnn_label, cnn_proba, cnn_classes, _ = predict_cnn(temp_audio_path)

            st.markdown("---")
            st.subheader("Tahmin sonuçları")

            col1, col2 = st.columns(2)

            with col1:
                xgb_prob_df = make_probability_df(xgb_classes, xgb_proba)
                st.markdown(f"### XGBoost: {emotion_text(xgb_label)}")
                st.pyplot(
                    plot_probability_chart(xgb_prob_df, "XGBoost Olasılıkları")
                )

            with col2:
                cnn_prob_df = make_probability_df(cnn_classes, cnn_proba)
                st.markdown(f"### CNN: {emotion_text(cnn_label)}")
                st.pyplot(
                    plot_probability_chart(cnn_prob_df, "CNN Olasılıkları")
                )

        except Exception as hata:
            st.error(f"Hata oluştu: {hata}")

        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    else:
        st.info("Tahmin yapmak için bir ses dosyası yükleyin.")


# 2. sekme : metrikler
with tab2:
    st.header("Model metrik karşılaştırması")

    xgb_metrics, cnn_metrics = load_metrics_data()

    if xgb_metrics is None or cnn_metrics is None:
        st.warning("Metrik dosyaları bulunamadı. Önce XGBoost ve CNN eğitimlerini çalıştırmalısın.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(
                plot_metrics_for_model("XGBoost", xgb_metrics)
            )

        with col2:
            st.pyplot(
                plot_metrics_for_model("CNN", cnn_metrics)
            )


# 3. sekme : gorseller
with tab3:
    st.header("Kayıtlı sonuç görselleri")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("XGBoost Confusion Matrix")
        if XGB_CM_PATH.exists():
            st.image(str(XGB_CM_PATH))
        else:
            st.warning("XGBoost confusion matrix bulunamadı.")

    with col2:
        st.subheader("CNN Confusion Matrix")
        if CNN_CM_PATH.exists():
            st.image(str(CNN_CM_PATH))
        else:
            st.warning("CNN confusion matrix bulunamadı.")

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("CNN Eğitim Grafiği")
        if CNN_TRAIN_GRAPH_PATH.exists():
            st.image(str(CNN_TRAIN_GRAPH_PATH))
        else:
            st.warning("CNN eğitim grafiği bulunamadı.")

    with col4:
        st.subheader("XGBoost Özellik Önemi")
        if XGB_FEATURE_GRAPH_PATH.exists():
            st.image(str(XGB_FEATURE_GRAPH_PATH))
        else:
            st.warning("XGBoost özellik önemi grafiği bulunamadı.")