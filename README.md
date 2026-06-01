# COVID-19 & Pneumonia X-Ray Classification: An Ablation Study

================ENG================

## 📌 Project Overview
This project aims to classify chest X-ray images into three categories: **COVID-19, Normal, and Viral Pneumonia**. The primary goal is not only to achieve high classification accuracy but also to conduct a comprehensive **Ablation Study** to evaluate the impact of paranchymal lung masking and contrast enhancement (CLAHE) on model performance. We compared classical Machine Learning baselines with State-of-the-Art (SOTA) Deep Learning architectures.

## 📊 Dataset
The models were trained and evaluated using the **COVID-19 Radiography Database**.
* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
* **Handling Imbalance:** The dataset suffers from severe class imbalance (e.g., 10k Normal vs. 1.3k Viral Pneumonia). We addressed this using **Undersampling** for Classical ML and **Strict Class Weight Penalties** for Deep Learning pipelines.

## 🏗️ Project Structure & Scripts
* `01_preprocessing.py`: Downloads the dataset, applies lung masks (background removal), enhances contrast using CLAHE, and prepares the "Masked" dataset.
* `01b_preprocessing_not_masked.py`: Prepares the "Not Masked" dataset for the ablation study by applying Global CLAHE and Unsharp Masking without removing the background (bones/equipment).
* `02_baseline_and_ml.py`: Classical Machine Learning pipeline utilizing HOG (Histogram of Oriented Gradients) feature extraction combined with SVM and Random Forest classifiers on a strictly balanced subset.
* `03_dl_pipeline_final.py`: The core Deep Learning pipeline. Automates the training of 6 different architectures across both masked and unmasked datasets using Transfer Learning and dynamic data augmentation.
* `04_evaluation.py`: Loads the saved model weights to generate Classification Reports (Precision, Recall, F1-Score) and Confusion Matrices for final reporting.

## 🧠 Architectures Used
Instead of relying solely on traditional networks, we utilized a modern spectrum of CNNs:
1. **ResNet50 & DenseNet201:** Strong traditional feature extractors.
2. **Proposed Hybrid Model:** Concatenates deep features from ResNet50 and DenseNet201 to capture both macro-opacities and micro-lesions simultaneously.
3. **Xception:** Utilized for its depthwise separable convolutions, making it computationally efficient while maintaining high accuracy.
4. **EfficientNetV2:** Chosen for its Fused-MBConv layers, offering faster training times and superior parameter efficiency on limited medical data.
5. **ConvNeXt (Tiny):** The ultimate modern CNN. Used as a highly stable, native alternative to Vision Transformers (ViT), bridging the gap between pure CNNs and attention-based mechanisms.

## 👨‍💻 Developers
* **Ece İrem Şişer:** Deep Learning Architectures, Feature Fusion, and Ablation Pipeline.
* **Doğukan Çakır:** Classical Machine Learning (HOG+SVM) Baselines and Data Balancing.

<br>

================TUR================

## 📌 Proje Özeti
Bu proje, göğüs röntgeni görüntülerini üç kategoriye ayırmayı amaçlamaktadır: **COVID-19, Normal ve Viral Pnömoni**. Temel amaç sadece yüksek sınıflandırma doğruluğu elde etmek değil, aynı zamanda parankimal akciğer maskeleme ve kontrast iyileştirmenin (CLAHE) model performansı üzerindeki etkisini değerlendirmek için kapsamlı bir **Ablasyon Çalışması (Etki Analizi)** yürütmektir. Klasik Makine Öğrenmesi algoritmaları ile modern Derin Öğrenme mimarileri karşılaştırmalı olarak incelenmiştir.

## 📊 Veri Seti
Modeller, **COVID-19 Radiography Database** kullanılarak eğitilmiş ve test edilmiştir.
* **Kaynak:** [Kaggle Veri Seti Linki](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
* **Dengesizlik Çözümü:** Veri setindeki ciddi sınıf eşitsizliği (Örn: 10 bin Normal'e karşı 1.3 bin Pnömoni), Makine Öğrenmesi için **Alt Örnekleme (Undersampling)**, Derin Öğrenme için ise **Sınıf Ağırlığı Cezalandırması (Class Weights)** yöntemleriyle çözülmüştür.

## 🏗️ Proje Yapısı ve Kodlar
* `01_preprocessing.py`: Veri setini indirir, akciğer maskelerini uygular (arka plan silinir), CLAHE ile kontrastı artırır ve "Maskeli" veriyi hazırlar.
* `01b_preprocessing_not_masked.py`: Ablasyon çalışması için "Maskesiz" veriyi hazırlar. Arka planı (kemikler/cihazlar) silmeden Global CLAHE ve Keskinleştirme uygular.
* `02_baseline_and_ml.py`: Dengelenmiş veri alt kümesi üzerinde HOG (Yönlü Gradyan Histogramı) özellik çıkarımı ile SVM ve Random Forest sınıflandırıcılarını kullanan Klasik Makine Öğrenmesi boru hattıdır.
* `03_dl_pipeline_final.py`: Ana Derin Öğrenme kodudur. 6 farklı mimarinin, Öğrenme Transferi ve dinamik veri artırımı kullanılarak hem maskeli hem maskesiz verilerde eğitimini otomatikleştirir.
* `04_evaluation.py`: Kaydedilen model ağırlıklarını yükleyerek Sınıflandırma Raporları (F1-Skoru) ve Karmaşıklık Matrisleri (Confusion Matrix) üretir.

## 🧠 Kullanılan Mimariler
1. **ResNet50 & DenseNet201:** Geleneksel, güçlü özellik çıkarıcı ağlar.
2. **Önerilen Hibrit Model:** Hem büyük lekelenmeleri hem de kılcal doku bozulmalarını aynı anda yakalamak için ResNet50 ve DenseNet201'in derin özellik havuzlarını birleştirir.
3. **Xception:** Derinliğine ayrılabilir evrişimleri (depthwise separable convolutions) sayesinde hesaplama maliyetini düşürürken doğruluğu koruduğu için kullanılmıştır.
4. **EfficientNetV2:** Fused-MBConv katmanları sayesinde tıbbi verilerde daha az parametre ile daha hızlı ve stabil öğrenme sağladığı için seçilmiştir.
5. **ConvNeXt (Tiny):** Vision Transformer'lara (ViT) karşı modern CNN'lerin cevabıdır. Transformer mekanizmalarının gücünü, klasik CNN mimarisinin stabilitesiyle sunduğu için modern bir referans olarak kullanılmıştır.

## 👨‍💻 Geliştiriciler
* **Ece İrem Şişer:** Derin Öğrenme Mimarileri, Hibrit Model (Feature Fusion) ve Ablasyon Çalışması.
* **Doğukan Çakır:** Klasik Makine Öğrenmesi (HOG+SVM) ve Veri Dengeleme Analizi.
