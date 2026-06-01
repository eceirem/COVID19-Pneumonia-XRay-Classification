# COVID-19 & Pnömoni Radyografi Sınıflandırması: Derin Öğrenme ve Ablasyon Çalışması

**Ankara Üniversitesi Mühendislik Fakültesi Bilgisayar Mühendisliği Bölümü — Sayısal Görüntü İşleme Dersi Projesi**

![Conference](https://img.shields.io/badge/Conference-IHCONCS_2026-4B0082)
![GPU](https://img.shields.io/badge/Hardware-NVIDIA_A100-76B900?logo=nvidia&logoColor=white)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow_%7C_Keras-FF6F00?logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white)

![Xception](https://img.shields.io/badge/Model-Xception-1f425f)
![DenseNet201](https://img.shields.io/badge/Model-DenseNet201-1f425f)
![ResNet50](https://img.shields.io/badge/Model-ResNet50-1f425f)
![ConvNeXt](https://img.shields.io/badge/Model-ConvNeXt__Tiny-1f425f)
![EfficientNetV2](https://img.shields.io/badge/Model-EfficientNetV2-1f425f)
![Hybrid](https://img.shields.io/badge/Proposed-Hybrid__ResNet__DenseNet-e11d48)

Bu repo, göğüs röntgeni (X-Ray) görüntülerini **COVID-19, Normal ve Viral Pnömoni** olmak üzere üç kategoriye ayırmayı amaçlayan uçtan uca bir yapay zeka boru hattını (pipeline) barındırmaktadır. 

Projenin temel bilimsel katkısı sadece yüksek doğruluk oranlarına ulaşmak değil; **parankimal akciğer maskeleme (segmentasyon)** ve **kontrast iyileştirme (CLAHE)** işlemlerinin model performansı üzerindeki etkilerini ölçen kapsamlı bir **Ablasyon Çalışması (Etki Analizi)** yürütmektir. Geleneksel Makine Öğrenmesi (ML) algoritmaları ile SOTA (State-of-the-Art) Derin Öğrenme (DL) mimarileri karşılaştırmalı olarak incelenmiştir.

---
## 👩‍💻 Proje Ekibi ve Araştırmacılar

Bu çalışma, veri mühendisliğinden model optimizasyonuna kadar titiz bir AR-GE süreciyle geliştirilmiştir.

### Geliştirici Kadrosu

* **Ece İrem ŞİŞER** [![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/eceiremsiser) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ece-irem-şişer/)
  * *Derin Öğrenme Mimarileri, Hibrit Model (Feature Fusion), A100 GPU Optimizasyonu ve Ablasyon Çalışması*

* **Doğukan ÇAKIR** [![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/dcakir01) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/doğukan-çakir-230076252/)
  * *Klasik Makine Öğrenmesi (HOG+SVM), Veri Dengeleme Analizi (Undersampling) ve Temel Çıkarımlar*

---

## 🔬 Ablasyon Çalışması ve "Maskeleme Paradoksu"

Tıbbi görüntü işlemede genel kabul gören *"arka plan gürültüsünü (kemikler, tıbbi cihazlar vb.) maskeleyerek silme"* varsayımı bu projede test edilmiştir. 

Modellerimiz hem **Maskeli (Sadece akciğer parankimi)** hem de **Maskesiz (Tüm röntgen + Global CLAHE)** veri setleri üzerinde eğitilmiştir. Deneyler sonucunda, literatüre katkı sağlayacak bir **"Maskeleme Paradoksu"** keşfedilmiştir: Maskelenmemiş verilerdeki radyolojik yansımaların (kemik yoğunluğu, mediastinal yapılar), modeller (özellikle Xception ve Logistic Regression) tarafından ayırt edici birer öznitelik (feature) olarak kullanıldığı ve maskesiz senaryoların bazı mimarilerde daha yüksek performans sunduğu kanıtlanmıştır.

### ⚙️ Veri Seti ve Dengesizlik Çözümü
Modeller, Kaggle **COVID-19 Radiography Database** kullanılarak eğitilmiştir. Veri setindeki ciddi sınıf dengesizliği (Örn: 10.000 Normal vs. 1.300 Pnömoni) şu şekilde çözülmüştür:
* **Makine Öğrenmesi (ML) İçin:** Rastgele Alt Örnekleme (Undersampling) ile azınlık sınıfı baz alınarak veri seti dengelenmiştir.
* **Derin Öğrenme (DL) İçin:** Veri kaybını önlemek adına dinamik Veri Artırımı (Data Augmentation) ve kayıp fonksiyonuna entegre edilen **Sınıf Ağırlığı Cezalandırması (Class Weights)** yöntemleri kullanılmıştır.

---

## 🧠 Kullanılan Mimariler ve Hibrit Yaklaşım

Projede geleneksel ağların yanı sıra parametre verimliliği yüksek modern CNN mimarileri kullanılmıştır:

1. **Önerilen Hibrit Model:** Hem büyük makro-lekelenmeleri (macro-opacities) hem de kılcal doku bozulmalarını (micro-lesions) aynı anda yakalamak için **ResNet50** ve **DenseNet201**'in derin özellik havuzlarını (feature maps) birleştiren (Concatenate) özgün bir mimari.
2. **Xception:** Derinliğine ayrılabilir evrişimleri (depthwise separable convolutions) sayesinde hesaplama maliyetini düşürürken doğruluğu koruyan lider modelimiz.
3. **ConvNeXt (Tiny):** Vision Transformer'lara (ViT) karşı modern CNN'lerin cevabı olarak, dikkat (attention) mekanizmalarının gücünü klasik CNN stabilitesiyle sunan referans mimari.
4. **EfficientNetV2:** Fused-MBConv katmanları sayesinde tıbbi verilerde parametre verimliliği sağlaması amacıyla teste dahil edilmiştir.
5. **ResNet50 & DenseNet201:** Güçlü, geleneksel özellik çıkarıcı (baseline) ağlar.

---

## 📂 Repo Mimarisi ve Dosya Yapısı

Proje, araştırma not defterleri, deneysel sonuçlar ve raporları barındıran modüler bir yapıya sahiptir. *(Not: Değerlendirme işlemleri (Evaluation) izolasyonu sağlamak amacıyla ayrı bir dosyada değil, eğitim pipeline'ı (`03`) içerisinde `validation_split` ve sabit `seed` mantığıyla uçtan uca kurgulanmıştır.)*

```text
COVID19-Pneumonia-Classification/
├── README.md                      # Ana dökümantasyon paneli
├── notebooks/                     # Araştırma ve Geliştirme Kodları
│   ├── 01_preprocessing.ipynb           # Akciğer maskeleme ve lokal CLAHE
│   ├── 01b_preprocessing_not_masked.ipynb # Maskesiz Global CLAHE ve Unsharp Masking
│   ├── 02_baseline_and_ml.ipynb         # HOG özellik çıkarımı ve SVM/LogReg
│   └── 03_dl_pipeline_final.ipynb       # 6 Mimarinin eğitimi ve Hold-out Testi (A100 GPU)
├── results/                       # Sayısal Performans Çıktıları
│   ├── ml_benchmarks/                   # Doğukan'ın ML sonuçları (CSV/Excel)
│   └── dl_benchmarks/                   # Ece'nin DL mimari sonuçları (Excel tabloları)
└── assets/                        # Görsel Analizler ve Raporlar
    ├── reports/                         # Vize ve Final Teslim Raporları (PDF/DOCX)
    └── confusion_matrices/              # 12 adet (Maskeli/Maskesiz) Karmaşıklık Matrisi Grafikleri
```
## 🛠️ Modüler Geliştirme Adımları (Kodların İşlevleri)

Projenin teknik olgunlaşma süreci aşağıdaki modüller üzerinden yürütülmüştür:

* **`01_preprocessing`:** Veri setini indirir, akciğer maskelerini uygular, arka planı siler ve "Maskeli" veri setini Colab'in lokal diskine çıkarır.
* **`01b_preprocessing_not_masked`:** Ablasyon çalışmasının karşıt argümanı olan "Maskesiz" veriyi hazırlar. Arka planı silmeden, **Global CLAHE ve Keskinleştirme (Unsharp Masking)** uygulayarak kemik ve doku kontrastını artırır.
* **`02_baseline_and_ml`:** Dengelenmiş veri alt kümesi üzerinde **HOG (Yönlü Gradyan Histogramı)** özellik çıkarımı yapar. Klasik makine öğrenmesi (SVM, Random Forest, Logistic Regression) modellerini eğiterek temel sınırları (baseline) belirler.
* **`03_dl_pipeline_final`:** Sistemin kalbidir. **NVIDIA A100 GPU** üzerinde `mixed_float16` hassasiyetiyle çalışır. 6 farklı mimarinin hem maskeli hem maskesiz verilerde eğitimini (Transfer Learning & Fine-Tuning) otomatikleştirir. Aynı dosya içerisinde `validation_split` kullanılarak izole edilmiş test seti üzerinden Sınıflandırma Raporları (F1-Skoru) ve Karmaşıklık Matrisleri (Confusion Matrix) üretilir.

---

## 🏆 Kapsamlı Deneysel Performans Sonuçları (Özet)

Projenin tam sonuçları `results/` klasörü altındaki Excel dökümlerinde ve `assets/confusion_matrices/` içerisindeki ısı haritalarında (heatmap) mevcuttur.

### 🧬 Derin Öğrenme (DL) Öne Çıkan Bulgular:
* En yüksek başarım, Maskesiz veri seti üzerinde **Xception** modeli ile **%94.26 Validation Accuracy** ve **0.9871 AUC** skorlarıyla elde edilmiştir.
* Kendi tasarımımız olan **Hybrid_ResNet_DenseNet** mimarisi, Maskeli verilerde **%92.15 Accuracy** ile olağanüstü bir stabilite sergilemiştir.

### ⚙️ Makine Öğrenmesi (ML) Öne Çıkan Bulgular:
* Özellik çıkarımının (feature extraction) gücünü kanıtlar nitelikte, Maskesiz veri seti üzerinde **Logistic Regression**, **%92.3**'lük doğruluğa ulaşarak birçok derin öğrenme mimarisini geride bırakacak kadar güçlü bir doğrusal ayrıştırılabilirlik (linear separability) göstermiştir.

---
> **🎯 Bilimsel Katkı ve Sonuç**
> 
> *Bu çalışma, tıbbi görüntüleme alanında yalnızca modellerin veri ön işleme (maskeleme/CLAHE) stratejilerine olan duyarlılığını kanıtlamakla sınırlı kalmamıştır. Proje kapsamında:*
> * *Modern Derin Öğrenme (DL) mimarileri ile Klasik Makine Öğrenmesi (HOG + ML) algoritmaları uçtan uca kıyaslanarak donanım maliyeti/başarım analizi yapılmış,*
> * *Makro-lekelenmeleri ve mikro-doku bozulmalarını aynı anda yakalamak üzere özgün bir **Hibrit (ResNet50 + DenseNet201)** mimari literatüre sunulmuş,*
> * *SOTA (State-of-the-Art) modellerin dengesiz medikal veri setlerindeki karakteristik tepkileri (örn: Mode Collapse) kapsamlı bir ablasyon çalışmasıyla haritalandırılmıştır.*
