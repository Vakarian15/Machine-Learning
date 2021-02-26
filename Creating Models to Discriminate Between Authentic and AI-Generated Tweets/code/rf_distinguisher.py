import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn import metrics
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
import statistics
import nltk
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# So we can actually see all columns of our training data set
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()  # Social media

popular_tags = ['NNP', 'NN', 'JJ', 'NNS', 'RB']     # Top 5 POS tags in corpus (across all data sets)


def read_tweets_csv(tweets_csv):
    data_df = pd.read_csv(tweets_csv,
                          header=None,
                          names=['TWEET_MSG', 'AUTHOR'],
                          sep=",",
                          dtype=str).dropna()

    data_df['NUM_HASHTAGS'] = data_df['TWEET_MSG'].str.count(re.compile('(#\\w*)'))
    data_df['NUM_MENTIONS'] = data_df['TWEET_MSG'].str.count(re.compile('(@\\w*)'))

    sentiment_cols = ["POS", "NEU", "NEG", "COMPOUND"]
    data_df[sentiment_cols] = data_df.apply(compute_sentiment_scores, axis=1)

    token_cols = ['TWEET_WORD_TOKENS', 'TWEET_PUNCTUATIONS']
    data_df[token_cols] = data_df['TWEET_MSG'].apply(get_words_from_tweet_msg)

    data_df['NUM_SENTENCES'] = data_df['TWEET_MSG'].apply(lambda tweet_msg: len(sent_tokenize(tweet_msg)))
    data_df['WORD_COUNT'] = data_df['TWEET_WORD_TOKENS'].apply(lambda words: get_word_count(words))

    data_df['AVG_SENTENTIAL_LENGTH'] = data_df.apply(get_average_sentential_length, axis=1)

    data_df['AVG_WORD_LENGTH_CHOICE'] = data_df['TWEET_WORD_TOKENS'].apply(get_average_word_length)

    # data_df[popular_tags] = data_df['TWEET_WORD_TOKENS'].apply(get_preferred_part_of_speech)
    data_df[popular_tags] = data_df.apply(get_preferred_part_of_speech, axis=1)


    return data_df


def get_average_sentential_length(data_row):
    if data_row['NUM_SENTENCES'] > 0:
        return data_row['WORD_COUNT'] / data_row['NUM_SENTENCES']

    return 0


def get_word_count(tweet_word_tokens):
    words = [word for word in tweet_word_tokens if word not in stop_words]

    return len(words)


def get_average_word_length(tweet_word_tokens):
    noiseless = [len(word) for word in tweet_word_tokens if word not in stop_words]

    if len(noiseless) > 0:
        return statistics.mean(noiseless)

    # Contains only stop words?
    noisy = [len(word) for word in tweet_word_tokens]

    if len(noisy) > 0:
        return statistics.mean(noisy)

    return 0


# Top 5 POS tags in corpus
# Noiseless Words only (in all data files): [('NNP', 2366), ('NN', 2292), ('JJ', 1920), ('NNS', 1566), ('RB', 1158)]
# Real tweets: [('NNP', 799), ('NN', 691), ('JJ', 608), ('NNS', 452), ('RB', 433),
#               ('PRP', 385), ('VBG', 357), ('VBD', 340), ('VBP', 326), ('DT', 238)]
# Fake tweets: [('NNP', 775), ('NN', 694), ('JJ', 605), ('NNS', 467), ('RB', 410),
#               ('PRP', 370), ('VBG', 340), ('VBD', 337), ('VBP', 337), ('DT', 216)]
def get_preferred_part_of_speech(data_row):
    tweet_word_tokens = data_row['TWEET_WORD_TOKENS']

    # Exclude meaningless words
    words = [word for word in tweet_word_tokens if word not in stop_words]

    tags = nltk.pos_tag(words)  # Tag words

    # Get count of each tag
    tag_fd = nltk.FreqDist(tag for (word, tag) in tags)

    pos_freq = dict(tag_fd)

    return pd.Series([(pos_freq[x]/data_row['WORD_COUNT'])*100 if pos_freq.get(x) is not None else 0 for x in popular_tags])


def get_clean_tweet(text):
    # Remove URLs, @usernames, and #hashtags
    result = re.sub(r"http\S+", "", text)

    result = re.sub("(@\\w*)", "", result)

    return re.sub("(#\\w*)", "", result)


def get_words_from_tweet_msg(tweet_msg):
    text = get_clean_tweet(tweet_msg)

    # Tokenize tweet message into words
    tokenized_word = word_tokenize(text)

    # Word only: remove . (, , "", &, ;, etc.
    words = [word for word in tokenized_word if word.isalnum()]

    # Everything else
    notwords = len(tokenized_word) - len(words)

    return pd.Series([words, (notwords/len(tweet_msg))*100])


def get_combined_dataset_with_features(data_files):
    data_frames = []
    for f in data_files:
        data_frames.append(read_tweets_csv(f))
    data = pd.concat(data_frames)
    return data


def compute_sentiment_scores(data_row):
    tweet_msg = get_clean_tweet(data_row['TWEET_MSG'])

    sentiments = analyzer.polarity_scores(tweet_msg)

    return pd.Series([sentiments["pos"] * 100,
                      sentiments["neu"] * 100,
                      sentiments["neg"] * 100,
                      sentiments["compound"] * 100])


def get_trained_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Useful features for this training set, or for this data set combination
    feature_importances = [(X_train.columns[i], model.feature_importances_[i]) for i in range(len(X_train.columns))]
    print(sorted(feature_importances, reverse=True, key=itemgetter(1)))

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy: {}".format(metrics.accuracy_score(y_test, y_pred)))
    print("Balanced Accuracy: {}".format(metrics.balanced_accuracy_score(y_test, y_pred)))
    # print("Classification Report:")
    # print(metrics.classification_report(y_test, y_pred))


# WORD_COUNT, TWEET_PUNCTUATIONS, AVG_WORD_LENGTH_CHOICE, COMPOUND, NEU
# Accuracy: 0.7025133910177174
# Balanced Accuracy: 0.7002734769145125
# data_files = ['data/5k_trump_tweets.csv', 'data/5k_fake_trump_tweets.csv']

# AVG_WORD_LENGTH_CHOICE, TWEET_PUNCTUATIONS, COMPOUND, NEU, WORD_COUNT
# Accuracy: 0.5917576611482916
# Balanced Accuracy: 0.5916579008273164
data_files = ['data/5k_teigen_tweets.csv', 'data/5k_fake_teigen_tweets.csv']

# NNP, AVG_WORD_LENGTH_CHOICE, TWEET_PUNCTUATIONS, WORD_COUNT, NN
# Accuracy: 0.9110644257703081
# Balanced Accuracy: 0.9108946347034204
# data_files = ['data/5k_trump_tweets.csv', 'data/5k_teigen_tweets.csv']

# NNP, AVG_WORD_LENGTH_CHOICE, TWEET_PUNCTUATIONS, WORD_COUNT, COMPOUND
# Accuracy: 0.8958073889580739
# Balanced Accuracy: 0.892836635084697
# data_files = ['data/5k_fake_trump_tweets.csv', 'data/5k_fake_teigen_tweets.csv']


# Get data set
data = get_combined_dataset_with_features(data_files)

print(data)

# Separate features and target
y = data['AUTHOR']
X = data.drop(['AUTHOR', 'TWEET_MSG', 'TWEET_WORD_TOKENS'], axis=1)

# Get training testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train our model
trained_model = get_trained_model(X_train, y_train)

# Evaluate Model
evaluate_model(trained_model, X_test, y_test)
