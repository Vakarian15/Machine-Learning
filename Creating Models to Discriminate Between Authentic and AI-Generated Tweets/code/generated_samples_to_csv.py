import csv


tweets = []
with open('data/5k_fake_teigen_tweets.txt', 'r') as text_file:
    tweet = ''
    for line in text_file:
        if "====================" in line:
            if tweet != '' and '<|' not in tweet and '|>' not in tweet:
                tweets.append(tweet[0:280])
            tweet = ''
        else:
            tweet = tweet.join(line.replace('\n', ''))

author = 'faketeigen'
with open('data/5k_fake_teigen_tweets.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, quotechar='"')
    for tweet in tweets:
        writer.writerow([tweet, author])
