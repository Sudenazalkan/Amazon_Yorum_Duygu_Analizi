from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# METİN ÖN İŞLEME
# Veriyi okutalım
df = pd.read_excel("C:/Users/sudea/PycharmProjects/AmazonCaseStudy/amazon.xlsx")
print(df.head())

# Review değişkeni üzerinde tüm harfleri küçük harfe çevirelim
df['Review'] = df['Review'].str.lower()
print(df['Review'])

# Noktalama işaretlerini çıkartalım
df['Review'] = df['Review'].str.replace('[^\w\s]','')
print(df['Review'])

# Yorumlarda bulunan sayısal ifadeleri çıkaralım
df['Review'] = df['Review'].str.replace('\d','')
print(df['Review'])

# Bilgi içermeyen kelimeleri (stopwords) çıkaralım
sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(word for word in str(x).split() if word not in sw))
print(df['Review'])

# 1000'den az geçen kelimeleri veriden çıkaralım
temp_df = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in temp_df))
print(df['Review'])

# Lemmatization işlemini uygulayalım (kelime köklerine ayırma)
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
print(df['Review'])

# METİN GÖRSELLEŞTİRME
# Review değişkeninin içerdiği kelimelerin frekanslarını hesaplayalım ve tf olarak kaydedelim
tf = df['Review'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words","tf"]
print(tf)

# tf değişkeninin değeri 500'den çok olanlara göre filtreleme işlemi yaparak barplot ile görselleştirme işlemini yapalım
print(tf[tf["tf"]>500])
tf[tf["tf"]>500].plot.bar(x='words',y='tf')
plt.show()

# Review değişkeninin içerdiği tüm kelimeleri text isminde string olarak kaydedelim
text = " ".join(i for i in df.Review)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# DUYGU ANALİZİ
sia = SentimentIntensityAnalyzer()
# İlk 10 gözlem için polarity_scores() hesaplayalım
first_10 = df.head(10).copy()  # İlk 10 satırı al
first_10['polarity_score'] = first_10['Review'].apply(lambda x: sia.polarity_scores(str(x)))
print(first_10[['Review', 'polarity_score']])

# İlk 10 gözlemin compound değerlerini gözlemleyelim
first_10['compound_score'] = first_10['Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
print(first_10[['Review', 'compound_score']])

# 10 gözlem için compound skorları 0'dan büyükse pos değilse neg şeklinde etiketleyelim
first_10['sentiment_label'] = first_10['Review'].apply(lambda x: "pos" if sia.polarity_scores(str(x))['compound'] > 0 else "neg")

# Şimdi "pos" ve "neg" etiketlerini 0-1 olarak kodlayalım
first_10['sentiment_label_encoded'] = LabelEncoder().fit_transform(first_10['sentiment_label'])

print(first_10[['Review', 'sentiment_label', 'sentiment_label_encoded']])

df['sentiment_label'] = df['Review'].apply(lambda x: "pos" if sia.polarity_scores(str(x))['compound'] > 0 else "neg")


train_x , test_x , train_y , test_y = train_test_split(df['Review'],
                                                       df['sentiment_label'],
                                                       random_state=42)
# TF-IDF Word Level
tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

# MODELLEME
# Lojistik regresyon modelini kurarak train dataları ile fit edelim
log_model = LogisticRegression().fit(x_train_tf_idf_word,train_y)

# Kurmuş olduğumuz modelle tahmin işlemleri gerçekleştirelim
from sklearn.metrics import classification_report
y_pred = log_model.predict(x_test_tf_idf_word)
print(classification_report(y_pred,test_y))

cross_val_score(log_model,x_test_tf_idf_word,test_y,cv=5).mean()

# sample fonksiyonu ile Review değişkeni içerisinden örneklem seçerek yeni bir değere atayalım.
# Elde ettiğimiz örneklemi modelin tahmin edebilmesi için CountVectorizer ile vektörleştirelim
# Vektörleştirdiğimiz örneklemi fit ve transform işlemlerini yaparak kaydedelim
# Örneklemi ve tahmin sonucunu ekrana yazdıralım

random_review = pd.Series(df['Review'].sample(1).values)
new_commend = CountVectorizer().fit(train_x).transform(random_review)
pred = log_model.predict(new_commend)
print(f'Review:  {random_review[0]} \n Prediction:  {pred}')
