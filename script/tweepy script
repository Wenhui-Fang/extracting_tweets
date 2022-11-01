import pandas as pd
import tweepy
import matplotlib.pyplot as plt
import seaborn as sns

# #### 1. Extract 100 Tweets in the past 7 days that @mention or hashtag Microsoft, Amazon, and Google using three separate API calls
# #### so that we have more tweets to work with.

# Instantiate Twitter API Client
client = tweepy.Client(bearer_token= tolen,
                      return_type="Response")

### TWEETS WITH PAGINATOR, get 100 tweets that mention Microsoft, are not rts, and are in english
tweets_msft = tweepy.Paginator(client.search_recent_tweets,
                          query="@Microsoft OR #Microsoft -is:retweet lang:en",
                         max_results=100,
                         ).flatten(limit=100) # Set to extract 100 tweets

### TWEETS WITH PAGINATOR, get 100 tweets that mention Amazon, are not rts, and are in english
tweets_amzn = tweepy.Paginator(client.search_recent_tweets,
                          query="@Amazon OR #Amazon -is:retweet lang:en",
                         max_results=100,
                         ).flatten(limit=100) # Set to extract 100 tweets

### TWEETS WITH PAGINATOR, get 100 tweets that mention Google, are not rts, and are in english
tweets_goog = tweepy.Paginator(client.search_recent_tweets,
                          query="@Google OR #Google -is:retweet lang:en",
                         max_results=100,
                         ).flatten(limit=100) # Set to extract 100 tweets


# #### 2. Parse the results and create a dataframe that contains the raw text of the tweets. Process the text so that only lowercase letters remain.

# create a dataframe using list comprehension for Microsoft
df_msft = pd.DataFrame({
    "tweet":[i.text for i in tweets_msft],
})

# create a dataframe using list comprehension for Amazon
df_amzn = pd.DataFrame({
    "tweet":[i.text for i in tweets_amzn],
})

# create a dataframe using list comprehension for Google
df_goog = pd.DataFrame({
    "tweet":[i.text for i in tweets_goog],
})

# a user_defined function used to clean text
def process_tweets(t):
    import re
    
    t = t.lower()
    t = re.sub("@[A-Za-z0-9]+", "", t) # remove handles
    t = re.sub("#[A-Za-z0-9]+", "", t) # remove hashtags
    t = re.sub(r"http\S+", "", t)      # remove links (anything that doesn't have a space after http)
    t = re.sub(r"www.\S+", "", t)      # remove links
    t = re.sub("[()!?]", "", t)        # remove punctuation
    t = re.sub("\[.*?\]", "", t)       # remove puncutation
    t = re.sub("[^a-z]", " ", t)       # remove anything that is not a letter
    return t

# use the previously defined function to clean the Microsoft dataframe
df_msft["tweet_clean"] = df_msft["tweet"].apply(process_tweets)
df_msft.head()

# use the previously defined function to clean the Amazon dataframe
df_amzn["tweet_clean"] = df_amzn["tweet"].apply(process_tweets)
df_amzn.head()

# use the previously defined function to clean the Google dataframe
df_goog["tweet_clean"] = df_goog["tweet"].apply(process_tweets)
df_goog.head()

# #### 3. Vectorize the raw text and determine the most frequent single word mentions in tweets for each company. Determine the most frequent two-word phrases mentioned.

# define a function that outputs the most frequent single word or two-word phrases by taking in 3 arguments:
# a dataframe, a min ngram and a max ngram

def tweets_vectorizer(df, ngram_min = 1, ngram_max = 1):
    
    import pandas as pd
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk import ngrams
    
    # Instantiates a vectorizer to vectorize the text (extract features) based on ngram parameters we specify
    v = CountVectorizer(stop_words="english",
                       ngram_range=(ngram_min, ngram_max))
    
    # Vectorizes the text
    dtm = v.fit_transform(df["tweet_clean"])
    
    # Creates a dataframe with terms and frequencies and converts the sparse arrays to dense arrays
    dtm_df = pd.DataFrame(dtm.todense(),
                     columns=v.get_feature_names())
    
    # transposes the dataframe, and sums the frequencies of the terms 
    dtm_df = dtm_df.T
    term_frequency = dtm_df.sum(axis = 1)
    
    # find the 5 most frequent words
    n_largest = term_frequency.nlargest(5)
    
    # creates a new dataframe with terms and frequencies
    df_n_largest = n_largest.to_frame().reset_index()
    df_n_largest.columns = ['term', 'frequency']
    
    # return the column of 5 most frequent words in a
    # data frame - the first element of the column term
    # is the top most frequent word
    return df_n_largest

# create dictionary that contains each company's name
# and its associated dataframe

df_dic = {
  "Microsoft": df_msft,
  "Amazon": df_amzn,
  "Google": df_goog
}

# for each dataframe that contains term and frequency,
# print the most frequent term
for key, value in df_dic.items():
    res = tweets_vectorizer(value)['term'][0]
    print(f'the most frequent single word for {key} is {res}')

# for each dataframe that contains two-word phrases and frequency,
# print the most frequent term

for key, value in df_dic.items():
    res = tweets_vectorizer(value, 2, 2)['term'][0]
    print(f'the most frequent two-word phrases for {key} is: {res}')

# #### 4. Create three plots showing the most frequent single word in the tweets about each company. 
#         Create three more plots showing the most frequent two or three word phrases in the tweets about each company.

# build a data frame that consists of two columns: term and frequency
# combine 3 plots into 3 for better visualization

# should we show the frequency?
# tweets_vectorizer(df_msft, 1, 1)[0]

# build a dictionary based on most frequnt term and frequency
dic_unigram = {
  "Microsoft": tweets_vectorizer(df_msft).iloc[0],
  "Amazon": tweets_vectorizer(df_amzn).iloc[0],
  "Google": tweets_vectorizer(df_goog).iloc[0]
}

# convert the previously created dictionary to a dataframe
# for single word term
df_unigram = pd.DataFrame.from_dict(dic_unigram, orient='index')
df_unigram

# use seaborn to plot the 3 most frequent single word for each company

sns.set_theme()
ax = sns.barplot(data=df_unigram, x="term", y="frequency", hue=df_unigram.index)
ax.set_title("The most frequent single Twitter word for Microsoft, Amazon and Google")
ax.set_xlabel("Term");

# build a dictionary based on most frequnt term and frequency
# for two-word phrases for each company

dic_bigram = {
  "Microsoft": tweets_vectorizer(df_msft, 2, 2).iloc[0],
  "Amazon": tweets_vectorizer(df_amzn, 2, 2).iloc[0],
  "Google": tweets_vectorizer(df_goog, 2, 2).iloc[0]
}


# convert the previously created dictionary to a dataframe
# for single word term

df_bigram = pd.DataFrame.from_dict(dic_bigram, orient='index')
df_bigram


# use seaborn to plot the 3 most frequent single word for each company

ax = sns.barplot(data=df_bigram, x="term", y="frequency", hue=df_bigram.index)
ax.set_title("The most frequent two-word Twitter phrases for Microsoft, Amazon and Google")
ax.set_xlabel("Term");


# #### 5. Define a user defined function that takes one argument--company name--and returns a dataframe that contains a column of clean (lower case, letters only) text of tweets that @mention or hashtag the company passed to the function.

def retrive_tweets(company):
    import pandas as pd
    import tweepy
    
    client = tweepy.Client(bearer_token=token,
                      return_type="Response")
    
    ### TWEETS WITH PAGINATOR, get 100 tweets that mention Microsoft, are not rts, and are in english
    tweets = tweepy.Paginator(client.search_recent_tweets,
                          query=f"@{company} OR #{company} -is:retweet lang:en",
                         max_results=100,
                         ).flatten(limit=100) # Set to extract 100 tweets
    
    # create a dataframe using list comprehension
    df = pd.DataFrame({"tweet":[i.text for i in tweets],})
    
    # use the previously defined function to clean tweets
    df["tweet_clean"] = df["tweet"].apply(process_tweets)
    
    return df[["tweet_clean"]]

retrive_tweets('Facebook')
