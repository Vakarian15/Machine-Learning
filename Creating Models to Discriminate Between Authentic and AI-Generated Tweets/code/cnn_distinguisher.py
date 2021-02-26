import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense 
from keras.optimizers import Adam

from string import punctuation
from nltk.corpus import stopwords

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def read_tweets_csv(tweets_csv):
    data_df = pd.read_csv(tweets_csv,
                          header=None,
                          names=['TWEET_MSG', 'AUTHOR'],
                          sep=",",
                          dtype=str).dropna()

    data_df['TWEET_MSG'] = data_df['TWEET_MSG'].apply(clean_tweets)

    return data_df

def clean_tweets(tweet):
    tokens = tweet.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens)
    return tokens

def get_combined_dataset(data_files):
    data_frames = []
    for f in data_files:
        data_frames.append(read_tweets_csv(f))
    data = pd.concat(data_frames)
    return data

def train_and_evaluate(X_train, X_test, y_train, y_test, vocab_size, max_length):

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)

    X_train = sequence.pad_sequences(X_train, maxlen=max_length, padding='post')
    X_test = sequence.pad_sequences(X_test, maxlen=max_length, padding='post')

    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=max_length))
    model.add(Dropout(0.5))
    model.add(Conv1D(32, 4, activation='relu'))
    model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(lr=0.0005)
    model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    model.fit(X_train, y_train,
          batch_size=32,
          epochs=30,
          validation_data=(X_test, y_test))


# List of csv files
data_files = ['data/5k_trump_tweets.csv', 'data/5k_fake_trump_tweets.csv']

# Get data set
data = get_combined_dataset(data_files)

# Separate features and target
y = data['AUTHOR']
X = data['TWEET_MSG']

# Get training testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

max_length=140
vocab_size=len(tokenizer.word_index) + 1

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Train and Evaluate the model
train_and_evaluate(X_train, X_test, y_train, y_test, vocab_size=vocab_size, max_length=max_length)
