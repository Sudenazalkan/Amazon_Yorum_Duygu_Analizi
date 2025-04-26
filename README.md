# 📦 Amazon Ürün Yorumları Üzerinden Duygu Analizi

Bu projede Amazon ürün yorumları üzerinde metin madenciliği ve duygu analizi çalışması gerçekleştirilmiştir.  
Proje, MIUUL tarafından verilen bir case study çalışması kapsamında tamamlanmıştır.

## İçerik

- **Metin Ön İşleme**
  - Küçük harfe çevirme
  - Noktalama işaretlerinin kaldırılması
  - Sayıların kaldırılması
  - Stopword'lerin çıkarılması
  - Nadir geçen kelimelerin çıkarılması
  - Lemmatization işlemi
  
- **Veri Görselleştirme**

📌 Word Cloud:
![Image](https://github.com/user-attachments/assets/008b2c6a-e35f-4fa5-86e6-58e9efb827eb)

📌 En Sık Geçen Kelimeler:
![Image](https://github.com/user-attachments/assets/a8d09072-a5cb-402e-8c41-29caf7bb3b43)

- **Duygu Analizi**
  - `SentimentIntensityAnalyzer` kullanılarak compound skorların hesaplanması
  - 0'dan büyük compound skoru: `positive (pos)`, diğerleri: `negative (neg)` olarak etiketlenmiştir.

- **Modelleme**
  - TF-IDF vektörleştirme
  - Logistic Regression modeli
  - Performans metriği olarak `accuracy`, `precision`, `recall`, `f1-score` kullanılmıştır.

## Sonuçlar

- Model doğruluk oranı (**accuracy**) yaklaşık **%89** olarak bulunmuştur.
- Pozitif yorumlarda model oldukça başarılı sonuçlar vermiştir.
- Negatif yorumların görece az olması, modelin negatif sınıf performansını bir miktar düşürmüştür.

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Negatif (neg) | 0.33      | 0.90   | 0.49     | 82      |
| Pozitif (pos) | 0.99      | 0.89   | 0.94     | 1321    |

### Genel Değerlendirme

- Pozitif yorumları tahmin etmede **yüksek başarı** sağlandı.
- Veri dengesizliğinden dolayı negatif yorumlar için **recall** değeri yüksek olmasına rağmen **precision** düşük kaldı.
- Veri setinin dengelenmesi veya farklı model denemeleri (örneğin SMOTE, XGBoost vb.) ile performans daha da artırılabilir.

## Kullanılan Kütüphaneler

- pandas
- numpy
- nltk
- sklearn
- matplotlib
- wordcloud

## Teşekkürler

Bu proje, MIUUL'un sunduğu eğitim materyalleri ve case study'lerinden ilham alınarak hazırlanmıştır.
