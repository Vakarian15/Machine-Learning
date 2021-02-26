import GetOldTweets3 as got
import csv


tweetCriteria = got.manager.TweetCriteria().setUsername("realDonaldTrump").setMaxTweets(10000)
author = 'trump'

tweets = got.manager.TweetManager.getTweets(tweetCriteria)

with open('data/10k_trump_tweets.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, quotechar='"')
    for tweet in tweets:
        if len(tweet.text) > 0:
            writer.writerow([tweet.text, author])
