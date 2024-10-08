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
df = pd.concat([df,df])

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
data['review'] = data['review'].apply(nfx.remove_punctuations)
data['review'] = data['review'].apply(nfx.remove_urls)
data['review'] = data['review'].apply(nfx.remove_multiple_spaces)
data['review'] = data['review'].apply(nfx.remove_currency_symbols)
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

!pip install nlpaug

import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src='wordnet')
for i in range(len(data)):
    text = ' '.join(data.loc[i, 'review'])
    label = data.loc[i, 'tag']
    augmented_text = aug.augment(text)
    augmented_text = ' '.join(augmented_text)
    data = data.append({'review': augmented_text.split(), 'tag': label}, ignore_index=True)

import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

nltk.download('stopwords')

# Tokenization
stop_words = set(stopwords.words('english'))
data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['review'])
X = tokenizer.texts_to_sequences(data['review'])
X = pad_sequences(X, padding='post', maxlen=100)

from sklearn.preprocessing import LabelEncoder

reviews = data['review'].values
labels = data['tag'].values
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

from sklearn.model_selection import train_test_split
train_sentences, test_sentences, train_labels, test_labels = train_test_split(reviews, encoded_labels, stratify = encoded_labels)

# Hyperparameters of the model
vocab_size = 3000 # choose based on statistics
oov_tok = ''
embedding_dim = 100
max_length = 200 # choose based on statistics, for example 150 to 200
padding_type='post'
trunc_type='post'
# tokenize sentences
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
# convert train dataset to sequence and pad sequences
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding='post', maxlen=max_length)
# convert Test dataset to sequence and pad sequences
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding='post', maxlen=max_length)

from keras.preprocessing.text import Tokenizer
import keras
# LSTM
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

num_epochs = 10
history = model.fit(train_padded, train_labels,
                    epochs=num_epochs, verbose=1,
                    validation_split=0.1)

from sklearn.metrics import accuracy_score

prediction = model.predict(test_padded)
# Get labels based on probability 1 if p>= 0.5 else 0
pred_labels = []
for i in prediction:
    if i >= 0.5:
        pred_labels.append(1)
    else:
        pred_labels.append(0)
print("Accuracy of prediction on test set : ", accuracy_score(test_labels,pred_labels))
