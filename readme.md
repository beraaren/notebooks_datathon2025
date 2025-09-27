# Datathon: Oturum Değeri Tahmin Modeli

Bu proje, bir e-ticaret platformundaki kullanıcı oturumlarının (session) gelecekteki potansiyel değerini tahmin etmeyi amaçlayan bir makine öğrenmesi modelini içermektedir. Proje, bir datathon kapsamında geliştirilmiş olup, kullanıcı davranışlarını analiz ederek anlamlı özellikler çıkarmayı ve bu özellikler üzerinden yüksek doğruluklu bir tahmin modeli oluşturmayı hedefler.

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
Projenin geliştirilmesinde aşağıdaki teknolojiler ve Python kütüphaneleri kullanılmıştır:
  * **Python 3.12**
  * **Pandas & NumPy**: Veri manipülasyonu ve analizi
  * **Matplotlib & Seaborn**: Veri görselleştirme
  * **Scikit-learn**: Veri ön işleme ve model değerlendirme
  * **CatBoost**: Gradyan artırma (gradient boosting) tabanlı makine öğrenmesi modeli
  * **SHAP**: Modelin tahminlerini açıklamak ve özelliklerin etkisini anlamak için
  * **Optuna**: Hiperparametre optimizasyonu
  * **Jupyter Notebook**: Analiz ve model geliştirme ortamı

### **Başarılı olan başlıca modeller**
ilk denemelerimiz neyi kullanıp kullanmayacağımıza karar vermek içindi. sonrasında uyguladığımız modellerin bir kısmı aşağıda belirtilmiştir.
bu modeller denediğimiz başarılı mimarilerin tamamını içerir.  
#### **v7 Modeli Özellikleri**

  * **Seans İçi Sıralama**: `event_order` (seans içindeki olay sırası) ve `event_order_pct` (olay sırasının seans uzunluğuna oranı) gibi özellikler eklenmiştir.
  * **Temel Etkileşim Oranları**: `view_to_add_cart_rate` (görüntülemeden sepete ekleme oranı) ve `add_cart_to_buy_rate` (sepete eklemeden satın alma oranı) gibi dönüşüm oranları hesaplanmıştır.
  * **Kullanıcı Bazlı Özellikler**: `user_total_events`, `user_lifespan_days`, `user_purchase_rate` gibi kullanıcının genel davranışını özetleyen özellikler türetilmiştir.
  * **Etkileşim Özellikleri**: `buy_x_hour` (satın alma sayısı ile ortalama saat etkileşimi) gibi özellikler, satın alma davranışını diğer metriklerle birleştirerek oluşturulmuştur.

#### **v8 Modeli Geliştirmeleri**

  * **Gelişmiş Zamansal İstatistikler**: Seans içindeki olaylar arası zaman farklarının (`time_diff`) standart sapması, medyanı ve logaritmik/karekök dönüşümleri gibi daha detaylı istatistikler eklenmiştir.
  * **İlk/Son Olay Özellikleri**: Bir seanstaki ilk ve son olayın türü, saati, ürünü ve kategorisi gibi bilgiler, seansın başlangıç ve bitiş dinamiklerini yakalamak için özellik olarak eklenmiştir.

#### **v13 Modeli **

  * **Özgün Etki Skorları**: Bir davranışsal imzanın (örneğin, VIEW -\> ADD\_CART -\> BUY) ortalama değerinden sapmayı (`delta`) hesaplayarak, kullanıcı ve kategorilerin "özgün etki" skorları türetilmiştir. Bu, modelin daha niş davranışları yakalamasına olanak tanımıştır.
  * **Meta-Modeller**: Kullanıcı ve kategori bazında daha küçük "meta-modeller" eğitilerek, bu modellerin tahminleri ana modele özellik olarak eklenmiştir. Bu, hiyerarşik bir öğrenme yaklaşımı sağlamıştır.
  * **Özellik Sentezi**: v7, v8 ve diğer yaklaşımlardan (`v16`, `v18`) elde edilen en iyi özellikler birleştirilerek nihai, kapsamlı bir özellik seti oluşturulmuştur.

## 📁 Jupyter Notebook'larının Açıklamaları

  * **`data_statistics_analysis.ipynb`**: Ham `train.csv` ve `test.csv` dosyaları üzerinde temel istatistiksel analizler, korelasyon matrisleri ve veri görselleştirmeleri yapar.
  * **`cat_00030_session_value_analysis.ipynb`**: Belirli bir kategori olan `CAT_00030` özelinde, olay sayısı, zaman ve ürün çeşitliliği gibi faktörlerin `session_value` üzerindeki etkisini derinlemesine inceler.
  * **`v7 modeli.ipynb`**: Özellik mühendisliğinin v7 versiyonunu ve CatBoost modelinin ilk temel eğitimini içerir.
  * **`v8 modeli.ipynb`**: v7 üzerine geliştirilmiş, daha gelişmiş zamansal ve sıralama özelliklerini içeren feature engineering adımlarını ve model eğitimini barındırır.
  * **`v13.ipynb`**: Projenin nihai ve en kapsamlı mimarisini içerir. v7, v8 ve diğer versiyonlardaki özellikleri birleştirir, meta-modeller eğitir ve son tahminleri yapar.
  * **`v7diğer_yaklaşımlar.ipynb`**: v7 modeline alternatif olarak geliştirilen veya ek olarak denenen farklı yaklaşımları ve analizleri içerir (örneğin, pseudo-labeling, gürültü simülasyonları).
  * **`shap_analysis.py`**: Eğitilmiş v8 modeli üzerinde SHAP analizi yaparak özelliklerin model tahminlerine olan etkisini görselleştirir ve açıklar.

-----
