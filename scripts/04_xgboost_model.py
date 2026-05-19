import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ana proje yolu
BASE_DIR = "proje/yolu"

#03 de kaydedilen veri yolu
features_path = os.path.join(BASE_DIR, "features", "03_features_xgb_clean.csv")

models_dir = os.path.join(BASE_DIR, "models")
outputs_dir = os.path.join(BASE_DIR, "outputs")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

# 03 de kaydedilen veriyi okuma
df = pd.read_csv(features_path)

print("Veri dosyası yüklendi.")
print(f"Kayıt sayısı   : {df.shape[0]}")
print(f"Sütun sayısı   : {df.shape[1]}")

# file ve emotion harici sutunlarin model ozellikleri
X = df.drop(columns=["file", "emotion"])
y_raw = df["emotion"]

# ozellik isimleri
feature_names = X.columns.tolist()

# guvenlik icin tum ozellikleri sayisal yapma
X = X.apply(pd.to_numeric, errors="coerce")

# eger eksik deger olursa sutun ortalamasiyla doldurma
if X.isnull().sum().sum() > 0:
    print("Eksik değer bulundu, sütun ortalamaları ile dolduruluyor...")
    X = X.fillna(X.mean())

# label encoding (etiketleri sayiya cevirme)
le = LabelEncoder()
y = le.fit_transform(y_raw)

# label encoder kaydetme
label_encoder_path = os.path.join(models_dir, "04_label_encoder_xgb.pkl")
joblib.dump(le, label_encoder_path)

print(f"\nÖzellik sayısı : {X.shape[1]}")
print(f"Sınıf sayısı   : {len(le.classes_)}")
print(f"Sınıflar       : {le.classes_}")

# sinif dagilimi
print("\nSınıf dağılımı:")
class_counts = pd.Series(y_raw).value_counts()

for emotion, count in class_counts.items():
    print(f"  {emotion:<15} → {count} kayıt")

class_counts.to_csv(
    os.path.join(outputs_dir, "04_xgb_class_distribution.csv"),
    encoding="utf-8-sig"
)

# train - test ayirma
X_train, X_test, y_train, y_test, file_train, file_test = train_test_split(
    X,
    y,
    df["file"],
    test_size=0.2, # %80 train %20 test
    random_state=42, # her seferinde ayni rastgelelik
    stratify=y # test train oranli dagilmasi
)

# ortak train-test dosya listesini kaydetme
split_df = pd.concat([
    pd.DataFrame({
        "file": file_train.values,
        "emotion": le.inverse_transform(y_train),
        "split": "train"
    }),
    pd.DataFrame({
        "file": file_test.values,
        "emotion": le.inverse_transform(y_test),
        "split": "test"
    })
], ignore_index=True)

split_df["stem"] = split_df["file"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

split_path = os.path.join(outputs_dir, "04_common_train_test_split.csv")
split_df.to_csv(split_path, index=False, encoding="utf-8-sig")

print(f"\nOrtak train-test split dosyası kaydedildi: {split_path}")

print(f"\nEğitim verisi : {len(X_train)} kayıt")
print(f"Test verisi   : {len(X_test)} kayıt")

print("\nEğitim sınıf dağılımı:")
for sinif, sayi in zip(*np.unique(y_train, return_counts=True)):
    print(f"  {le.classes_[sinif]:<15} → {sayi} kayıt")

print("\nTest sınıf dağılımı:")
for sinif, sayi in zip(*np.unique(y_test, return_counts=True)):
    print(f"  {le.classes_[sinif]:<15} → {sayi} kayıt")

# pipeline : scaler,smote,xgboost
pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
        tree_method="hist"
    ))
])

# hiperparametreler   (randomizedsearch icin)
param_dist = {
    "model__n_estimators": [200, 400, 600, 800],
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__learning_rate": [0.005, 0.01, 0.05, 0.1],
    "model__subsample": [0.6, 0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "model__min_child_weight": [1, 3, 5],
    "model__gamma": [0, 0.1, 0.3, 0.5],
    "model__reg_alpha": [0, 0.01, 0.1],
    "model__reg_lambda": [1, 1.5, 2]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=50,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1,
    refit=True,
    error_score="raise"
)

print("\nHiperparametre optimizasyonu başlıyor...")
search.fit(X_train, y_train)

best_pipeline = search.best_estimator_
best_model = best_pipeline.named_steps["model"]
best_scaler = best_pipeline.named_steps["scaler"]

print("\nEn iyi parametreler:")
print(search.best_params_)

print(f"\nEn iyi CV F1-Macro skoru: %{search.best_score_ * 100:.2f}")

# test verisi uzerinde degerlendirme
y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n--- TEST SONUÇLARI ---")
print(f"Accuracy        : %{accuracy * 100:.2f}")
print(f"Precision Macro : %{precision_macro * 100:.2f}")
print(f"Recall Macro    : %{recall_macro * 100:.2f}")
print(f"F1-Macro        : %{f1_macro * 100:.2f}")
print(f"F1-Weighted     : %{f1_weighted * 100:.2f}")

print("\n--- SINIFLANDIRMA RAPORU ---\n")
report_text = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_,
    zero_division=0
)

print(report_text)

# raporlari kaydetme
# 1. classification report CSV
report_dict = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

classification_report_path = os.path.join(
    outputs_dir,
    "04_xgb_classification_report.csv"
)

report_df.to_csv(classification_report_path, encoding="utf-8-sig")

# 2. ozet metrikler CSV
metrics_summary = pd.DataFrame([{
    "model": "XGBoost",
    "accuracy": accuracy,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
    "precision_weighted": precision_weighted,
    "recall_weighted": recall_weighted,
    "f1_weighted": f1_weighted,
    "best_cv_f1_macro": search.best_score_
}])

metrics_summary_path = os.path.join(
    outputs_dir,
    "04_xgb_metrics_summary.csv"
)

metrics_summary.to_csv(metrics_summary_path, index=False, encoding="utf-8-sig")

# 3. en iyi parametreleri kaydetme
best_params_clean = {
    key.replace("model__", ""): value
    for key, value in search.best_params_.items()
}

best_params_path = os.path.join(
    outputs_dir,
    "04_xgb_best_params.json"
)

with open(best_params_path, "w", encoding="utf-8") as f:
    json.dump(best_params_clean, f, ensure_ascii=False, indent=4)

# 4. test tahminlerini kaydetme
predictions_df = pd.DataFrame({
    "file": file_test.values,
    "true_label": le.inverse_transform(y_test),
    "predicted_label": le.inverse_transform(y_pred),
    "confidence": np.max(y_proba, axis=1)
})

for i, class_name in enumerate(le.classes_):
    predictions_df[f"prob_{class_name}"] = y_proba[:, i]

predictions_path = os.path.join(
    outputs_dir,
    "04_xgb_test_predictions.csv"
)

predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
).plot(
    ax=ax,
    colorbar=True,
    cmap="Blues"
)

ax.set_title("XGBoost - Confusion Matrix")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

cm_path = os.path.join(outputs_dir, "04_xgb_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.show()

print(f"\nConfusion matrix kaydedildi: {cm_path}")

# ozelliklerin onemi
importances = best_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

feature_importance_path = os.path.join(
    outputs_dir,
    "04_xgb_feature_importance.csv"
)

feature_importance_df.to_csv(feature_importance_path, index=False, encoding="utf-8-sig")

top_20 = feature_importance_df.head(20)

plt.figure(figsize=(10, 7))
plt.barh(
    top_20["feature"][::-1],
    top_20["importance"][::-1]
)

plt.title("XGBoost - En Önemli 20 Özellik")
plt.xlabel("Önem Skoru")
plt.ylabel("Özellik")
plt.tight_layout()

feature_importance_plot_path = os.path.join(
    outputs_dir,
    "04_xgb_feature_importance.png"
)

plt.savefig(feature_importance_plot_path, dpi=300)
plt.show()

print(f"Özellik önemi grafiği kaydedildi: {feature_importance_plot_path}")

# xgb modelini kaydetme
pipeline_path = os.path.join(models_dir, "04_xgb_pipeline.pkl")
model_path = os.path.join(models_dir, "04_xgb_model.pkl")
scaler_path = os.path.join(models_dir, "04_scaler_xgb.pkl")

joblib.dump(best_pipeline, pipeline_path)
joblib.dump(best_model, model_path)
joblib.dump(best_scaler, scaler_path)

print("\n✅ XGBoost modeli başarıyla kaydedildi.")
print(f"Pipeline          : {pipeline_path}")
print(f"Model             : {model_path}")
print(f"Scaler            : {scaler_path}")
print(f"Label Encoder     : {label_encoder_path}")

print("\n✅ Oluşturulan çıktı dosyaları:")
print(f"Classification report   : {classification_report_path}")
print(f"Metrik özeti            : {metrics_summary_path}")
print(f"En iyi parametreler     : {best_params_path}")
print(f"Test tahminleri         : {predictions_path}")
print(f"Confusion matrix        : {cm_path}")
print(f"Feature importance      : {feature_importance_path}")
print(f"Common train-test split : {split_path}")