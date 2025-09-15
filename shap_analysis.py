# --- 1. Gerekli Kütüphaneler ---
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import shap

print("İş akışı başladı: Veri Yükleme -> Model Eğitimi -> SHAP Analizi")

# --- 2. Veri Yükleme ve Hazırlama ---

# Modelini eğitirken kullandığın işlenmiş veriyi yükle
IN_TRAIN_PATH = "/content/datathon/processed/train_processed_v8.csv"
try:
    df_train = pd.read_csv(IN_TRAIN_PATH, index_col='user_session')
    print(f"'{IN_TRAIN_PATH}' verisi başarıyla yüklendi.")
except FileNotFoundError:
    print(f"Hata: '{IN_TRAIN_PATH}' dosya yolunda bulunamadı. Lütfen kontrol edin.")
    exit()

# Özellikleri (X) ve hedefi (y) ayır
y = df_train['session_value']
X = df_train.drop(['session_value'], axis=1)

# Hedef değişkene log dönüşümü
y_log = np.log1p(y)

# Zaman bazlı doğrulama (Time-Based Validation)
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.2, shuffle=False # shuffle=False zaman serisi doğrulama için kritik!
)
print(f"Veri setleri oluşturuldu. Train: {X_train.shape[0]} satır, Validation: {X_val.shape[0]} satır.")

# --- 3. Model Parametreleri ve Eğitim ---

# Eğitim kodundan gelen en iyi hiperparametreler
best_params = {
    'learning_rate': 0.05705622600719216,
    'depth': 4,
    'l2_leaf_reg': 2.768927236825974,
    'colsample_bylevel': 0.8234334604424713,
    'min_child_samples': 56,
    'objective': 'RMSE',
    'random_seed': 42,
    'verbose': 500
}

# Kategorik özelliklerin indekslerini bulma (eğitim koduyla aynı mantık)
categorical_features_indices = [
    i for i, col in enumerate(X.columns)
    if col in ['first_event_category', 'first_event_product', 'first_event_type',
               'last_event_category', 'last_event_product', 'last_event_type',
               'is_weekend', 'time_of_day']
]
print(f"CatBoost için kategorik özellik indeksleri: {categorical_features_indices}")

# CatBoost modelini tanımla
cat_model = CatBoostRegressor(
    **best_params,
    iterations=4500,
    eval_metric='RMSE',
    early_stopping_rounds=300,
    cat_features=categorical_features_indices
)

print("\nModel eğitimi başlıyor...")
# Modeli eğit
cat_model.fit(
    X_train, y_train_log,
    eval_set=(X_val, y_val_log)
)
print("Model eğitimi tamamlandı.")

# --- 4. SHAP ANALİZİ ---
print("\nSHAP değerleri hesaplanıyor...")

# Ağaç tabanlı modeller için TreeExplainer kullanılır
# Modeli dosyadan yüklemek yerine, az önce eğittiğimiz 'cat_model' değişkenini doğrudan kullanıyoruz
explainer = shap.TreeExplainer(cat_model)

# SHAP değerlerini hesapla
shap_values = explainer.shap_values(X_val)
print("SHAP değerleri başarıyla hesaplandı.")


# --- 5. GÖRSELLEŞTİRME ---

# Jupyter/Colab ortamında JS'i başlat (görseller için)
shap.initjs()

# Tek bir tahminin (validasyon setinin ilk satırı) analizini yap
print("\nİlk satır için Force Plot:")
display(shap.force_plot(explainer.expected_value, shap_values[0,:], X_val.iloc[0,:]))

# Modelin genel davranışını analiz et
print("\nModelin genel davranışını gösteren Summary Plot:")
shap.summary_plot(shap_values, X_val)