pip install neattext

import pandas as pd
import numpy as np
import seaborn as sns

df1 = pd.read_csv('/content/amazon.csv',header = None)
df2 = pd.read_csv('/content/imdb.csv',header = None)
df3 = pd.read_csv('/content/yelp.csv',header = None)

print(df1.shape)
print(df2.shape)
print(df3.shape)

df1[1].isna().sum()
df2[1].isna().sum()
df3[1].isna().sum()
df2.dropna(axis=0, inplace=True)
df2[1].isna().sum()
df1.head()
df2.head()

dict(df2.dtypes)[1]

df2[1]

df2[1] = df2[1].astype(int)
df2.head()

df3.head()

df = pd.concat([df1, df2, df3])

df.shape

df.head()

df[1].value_counts()

sns.countplot(x=df[1])

import neattext.functions as nfx

dir(nfx)

data = pd.DataFrame()
data['review'] = df[0]
data['tag'] = df[1]
data.head()

data['review'] = df[0].apply(nfx.remove_userhandles)
data['review'] = data['review'].apply(nfx.remove_stopwords)
data['review'] = data['review'].apply(nfx.remove_punctuations)
data['review'] = data['review'].apply(nfx.remove_urls)
data['review'] = data['review'].apply(nfx.remove_multiple_spaces)
data['review'] = data['review'].apply(nfx.remove_currency_symbols)
data['review'] = data['review'].apply(nfx.remove_shortwords)
data['review'] = data['review'].apply(nfx.remove_emojis)
data['review'] = data['review'].apply(nfx.remove_dates)
data['review'] = data['review'].apply(nfx.remove_phone_numbers)
data['review'] = data['review'].apply(nfx.remove_accents)
data['review'] = data['review'].apply(nfx.remove_hashtags)
data['review'] = data['review'].apply(nfx.remove_html_tags)
data['review'] = data['review'].apply(nfx.remove_non_ascii)
data['review'] = data['review'].apply(nfx.remove_puncts)
data['review'] = data['review'].apply(nfx.remove_special_characters)

data.head()

data['review'] = data['review'].str.lower()
data.head()

import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense

nltk.download('stopwords')

# Tokenization
stop_words = set(stopwords.words('english'))
data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['review'])
X = tokenizer.texts_to_sequences(data['review'])
X = pad_sequences(X, padding='post', maxlen=100)

y = np.array(data['tag'])
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
num_validation_samples = int(0.1 * X.shape[0])
X_train = X[:-num_validation_samples]
y_train = y[:-num_validation_samples]
X_test = X[-num_validation_samples:]
y_test = y[-num_validation_samples:]

X



#CNN
model = Sequential()
model.add(Embedding(5000, 32, input_length=100))
model.add(Conv1D(32, 7, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(32, 7, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
