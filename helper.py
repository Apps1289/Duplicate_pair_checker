import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('stopwords')

import re
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import paired_cosine_distances
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
model = pickle.load(open('w2v.pkl','rb'))




def test_common_words(q1,q2):
  q1 = set([word.lower() for word in q1.split()])
  q2 = set([word.lower() for word in q2.split()])
  return len(q1 & q2)
def test_total_words(q1,q2):
  q1 = set([word.lower() for word in q1.split()])
  q2 = set([word.lower() for word in q2.split()])
  return len(q1) + len(q2)
# Advance features

def test_fetch_token_features(q1,q2):
  SAFE_DIV = 0.0001

  token_features = [0.0]*8

  q1_tokens = q1.split()
  q2_tokens = q2.split()
  if len(q1_tokens) == 0 or len(q2_tokens) == 0:
    return token_features

  #non stopwords
  q1_words = set([word for word in q1_tokens if word not in stop_words])
  q2_words = set([word for word in q1_tokens if word not in stop_words])

  #stop_words
  q1_stops = set([word for word in q1_tokens if word in stop_words])
  q2_stops = set([word for word in q1_tokens if word in stop_words])

  #number  of  common words
  common_word_count = len(q1_words.intersection(q2_words))

  #number of common stop words
  stop_words_count = len(q1_stops.intersection(q2_stops))

  #number of common tokens
  common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

  token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
  token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) +  SAFE_DIV)
  token_features[2] = stop_words_count / (min(len(q1_stops), len(q2_stops))+ SAFE_DIV)
  token_features[3] = stop_words_count / (max(len(q1_stops), len(q2_stops))+ SAFE_DIV)
  token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens))+ SAFE_DIV)
  token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens))+ SAFE_DIV)
  token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
  token_features[7] = int(q1_tokens[0] == q2_tokens[0])

  return token_features



def test_fetch_length_features(q1,q2):


  q1_tokens = q1.split()
  q2_tokens = q2.split()

  length_features = [0.0]*3

  if len(q1_tokens) == 0 or len(q2_tokens) == 0:
    return length_features


  # mean_len
  length_features[0] = ((len(q1_tokens) / len(q2_tokens))/2)

  #abs_len_diff
  length_features[1] = abs(len(q1_tokens) - len(q2_tokens))

  #longest_common_subr
  strs = list(distance.lcsubstrings(q1,q2))
  length_features[2] = len(strs[0]) / (min(len(q1), len(q2))+1)

  return length_features


def test_fetch_fuzzy_features(q1,q2):



    fuzzy_features = [0.0]*4

    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features

def preprocess(q):
  q = str(q).lower().strip()

  q = q.replace('%', ' percent')
  q = q.replace('$', ' dollar ')
  q = q.replace('₹', ' rupee ')
  q = q.replace('€', ' euro ')
  q = q.replace('@', ' at ')

  q = q.replace(',000,000,000 ', 'b ')
  q = q.replace(',000,000 ', 'm ')
  q = q.replace(',000 ', 'k ')
  q = re.sub(r'([0-9]+)000000000', r'\1b', q)
  q = re.sub(r'([0-9]+)000000', r'\1m', q)
  q = re.sub(r'([0-9]+)000', r'\1k', q)

  contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
     "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }
  q_decontracted = []

  for word in q.split():
    if word in contractions:
      word = contractions[word]

    q_decontracted.append(word)

  q = ' '.join(q_decontracted)
  q = q.replace("'ve", " have")
  q = q.replace("n't", " not")
  q = q.replace("'re", " are")
  q = q.replace("'ll", " will")

  soup = BeautifulSoup(q)
  q = soup.get_text()

  pattern = re.compile('\W')
  q = re.sub(pattern, ' ', q).strip()

  return q

def extended_preprocess(q):
  q = q.lower().strip()

  # remove stopwords
  
  words = word_tokenize(q)
  filtered_words = [word for word in words if word.casefold() not in stop_words]
  q = ' '.join(filtered_words)

  #lemmatization
  lematizer = WordNetLemmatizer()
  words = word_tokenize(q)
  lemmatized_words = [lematizer.lemmatize(word) for word in words]
  q = ' '.join(lemmatized_words)

  return q

def get_sent_vec(sent,model):
  vectors = [model.wv[word] for word in sent if word in model.wv]
  if len(vectors) == 0:
    return np.zeros(300)
  else:
    return np.mean(vectors, axis=0)

def query_point(q1,q2,model):
  intent_query=[]

  q1 = preprocess(q1)
  q2 = preprocess(q2)

  #fetch basic features
  intent_query.append(len(q1))
  intent_query.append(len(q2))
  intent_query.append(len(q1.split()))
  intent_query.append(len(q2.split()))
  intent_query.append(test_common_words(q1,q2))
  intent_query.append(test_total_words(q1,q2))
  intent_query.append(round(test_common_words(q1,q2)/test_total_words(q1,q2),2))

  #fecth token features
  token_features = test_fetch_token_features(q1,q2)
  intent_query.extend(token_features)

    # fetch length based features
  length_features = test_fetch_length_features(q1,q2)
  intent_query.extend(length_features)

    # fetch fuzzy features
  fuzzy_features = test_fetch_fuzzy_features(q1,q2)
  intent_query.extend(fuzzy_features)

  q1 = extended_preprocess(q1)
  q2 = extended_preprocess(q2)

  v1 = get_sent_vec(simple_preprocess(q1), model).reshape(1, -1)
  v2 = get_sent_vec(simple_preprocess(q2), model).reshape(1, -1)

  cosine_sim = 1 - paired_cosine_distances(v1, v2)[0]


# Manhattan Distance (L1 Norm)
  manhattan_dist = np.linalg.norm(v1 - v2, axis=1, ord=1)

# Euclidean Distance (L2 Norm)
  euclidean_dist = np.linalg.norm(v1 - v2, axis=1, ord=2)

  features_basic = np.array(intent_query).reshape(1, -1)
  distances = np.array([cosine_sim, manhattan_dist.item(), euclidean_dist.item()]).reshape(1, -1)

  return np.hstack((features_basic, distances, v1, v2))