
from sklearn.model_selection import train_test_split
import xgboost as xgb

from scipy.sparse import hstack


import pandas as pd
import time
import nltk
import re
import string
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from typing import List, Optional



''' upload the dataset''' 
data = pd.read_csv('C:/Users/hp/Desktop/DisasterTweet/DisasterTweet/DisasterTweet/train.csv')





""" Supprime les abbreviation"""
class TweetPreprocessor:
    def __init__(self):
        self.known_abbreviations = {
            # Basic Internet Slang
            "u": "you","pls": "please","bc": "because","idk": "i do not know","omg": "oh my god","btw": "by the way","fyi": "for your information","ttyl": "talk to you later","lmao": "laughing my ass off","brb": "be right back","smh": "shaking my head","np": "no problem","cya": "see you","gtg": "got to go","lol": "laugh out loud","l8r": "later","gr8": "great","bff": "best friends forever","bffl": "best friends for life","fomo": "fear of missing out","yolo": "you only live once",
            # Extended Internet Expressions
            "omfg": "oh my f**king god","ftw": "for the win","b4": "before","xoxo": "hugs and kisses","hmu": "hit me up","glhf": "good luck have fun","rdy": "ready","plz": "please","ttfn": "ta ta for now","rofl": "rolling on the floor laughing","tbh": "to be honest","tbf": "to be fair","wbu": "what about you","lmk": "let me know","cu": "see you","bday": "birthday","bbl": "be back later","oic": "oh i see","yasss": "yes","lmfao": "laughing my f**king ass off",
            # Social Media Terms
            "bruh": "bro","fam": "family close friends","bae": "before anyone else","swag": "confident cool style","nm": "not much","tmi": "too much information","r": "are","imho": "in my humble opinion","woot": "expression of joy or excitement","ik": "i know","omw": "on my way","gr8t": "great","roflmao": "rolling on the floor laughing my ass off",
            # Modern Slang
            "wtf": "what the f**k", "wtbs": "whatever the bullshit", "lolz": "laugh out loud variation", "g2g": "got to go", "u2": "you too", "rly": "really", "omgosh": "oh my gosh", "ymmv": "your mileage may vary", "sis": "sister", "bro": "brother", "squad": "group of friends", "lit": "cool awesome", "salty": "being upset or angry", "sav": "savage", "flex": "show off", "no cap": "honestly no exaggeration",

            # Common Actions
            "h8": "hate", "jk": "just kidding", "nvm": "never mind", "gonna": "going to", "wanna": "want to", "gimme": "give me", "yass": "yes enthusiastic", "l8": "late", "lolol": "laugh out loud", "lms": "like my status",
            # Practical Abbreviations
            "j/w": "just wondering", "ppl": "people", "msg": "message", "msgs": "messages", "thx": "thanks", "ty": "thank you", "pic": "picture", "pics": "pictures", "prob": "probably", "rn": "right now",
            # With Symbols
            "w/": "with", "w/o": "without", "dm": "direct message", "rt": "retweet", "fb": "facebook",
            # Time Related
            "hrs": "hours", "hr": "hour", "min": "minute", "mins": "minutes", "sec": "second", "secs": "seconds", "am": "morning", "pm": "evening", "tmr": "tomorrow", "tmrw": "tomorrow",
            # Miscellaneous
            "tho": "though", "k": "thousand", "vs": "versus", "etc": "etcetera", "apt": "apartment", "apts": "apartments", "bldg": "building", "yr": "year", "yrs": "years", "ur": "your", "asap": "as soon as possible", "faq": "frequently asked questions",
            # Emergency Related
            "sos": "help", "emg": "emergency","evac": "evacuate",
        }

        self.numeric_abbreviations = {
            "2": "to","4": "for"
        }

    def _reduce_lengthening(self, text: str) -> str:
        pattern = re.compile(r'(.)\1{2,}')
        return pattern.sub(r'\1\1', text)

    def _tokenize(self, text: str) -> List[str]:
        pattern = r"""(?x)
            (?:[A-Za-z]\.)+         # Abbreviations with periods
          | [@#]?\w+(?:[-']\w+)*    # Words, hashtags, mentions
          | \$?\d+(?:\.\d+)?%?      # Numbers and percentages
          | \.\.\.+                 # Ellipsis
          | [][.,;"'?():-_`]        # Punctuation
        """
        return re.findall(pattern, text)

    def _handle_abbreviation(self, word: str) -> str:
        word_lower = word.lower()

        if word_lower in self.numeric_abbreviations:
            return self.numeric_abbreviations[word_lower]

        return self.known_abbreviations.get(word_lower, word)

    def _normalize_numbers(self, text: str) -> str:
        # Protection des abréviations numériques
        for num in self.numeric_abbreviations:
            text = re.sub(r'\b' + num + r'\b', f'NUM_ABBREV_{num}', text)

        # Normalisation standard des nombres
        text = re.sub(r'\d+%', 'PERCENTAGE', text)
        text = re.sub(r'\d+k\b', 'THOUSAND', text)
        text = re.sub(r'\$\d+', 'MONEY', text)
        text = re.sub(r'\b\d+\b', 'NUMBER', text)

        # Restauration des abréviations numériques
        for num in self.numeric_abbreviations:
            text = text.replace(f'NUM_ABBREV_{num}', num)

        return text

    def preprocess(self, text: Optional[str]) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower()

        # Traitement des éléments spéciaux
        text = re.sub(r'http\S+|www\S+', 'URL', text)
        text = re.sub(r'@\w+', 'USER', text)
        text = re.sub(r'#(\w+)', r'\1', text)

        # Protection et normalisation des nombres
        text = self._normalize_numbers(text)

        # Réduction des répétitions de caractères
        text = self._reduce_lengthening(text)

        # Tokenization et traitement des abréviations
        words = self._tokenize(text)
        words = [self._handle_abbreviation(word) for word in words]

        # Nettoyage final en préservant les apostrophes
        punctuation = string.punctuation.replace("'", "")
        words = [word for word in words if word not in punctuation]

        # Reconstruction du texte
        text = ' '.join(words)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()
def add_processed_column():
    global data
    preprocessor = TweetPreprocessor()
    data['processed_text'] = data['text'].apply(preprocessor.preprocess)
add_processed_column()





'''remove stop words'''
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    words = text.split()  # Split the text into words
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(filtered_words)  # Rebuild the text without stopwords
data['processed_text_no_stopwords'] = data['processed_text'].apply(remove_stopwords)
data['keyword'] = data['keyword'].fillna('unknown')
data['location'] = data['location'].fillna('unknown')


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
vectorizer = TfidfVectorizer(max_features=5000)  # You can tune the max_features parameter
X = vectorizer.fit_transform(data['processed_text_no_stopwords'])

y = data['target']  

# Split into train and test sets
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""XGBOOST"""
model_xgb = xgb.XGBClassifier(scale_pos_weight=2)  
model_xgb.fit(X_train_tfidf, y_train)
#y_pred_xgb = model_xgb.predict(X_test_tfidf)
import pickle

with open('xgboost.pkl', 'wb') as model_file:
    pickle.dump((model_xgb, vectorizer), model_file)
