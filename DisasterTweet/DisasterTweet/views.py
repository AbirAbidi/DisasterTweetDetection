from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import pickle  # Use pickle instead of joblib
import pandas as pd
import numpy as np
import time
import nltk
import re
import string
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from typing import Optional, List
import json

# Load the model and vectorizer using pickle
log_reg_model_path = 'C:/Users/hp/Desktop/DisasterTweet/log_reg_and_vectorizer.pkl'

with open(log_reg_model_path, 'rb') as model_file:
    log_reg_model, log_reg_vectorizer = pickle.load(model_file)  # Load model and vectorizer
    
svm_model_path = 'C:/Users/hp/Desktop/DisasterTweet/svm.pkl'

with open(svm_model_path, 'rb') as model_file:
    svm_model, svm_vectorizer = pickle.load(model_file) 
    
    
random_model_path = 'C:/Users/hp/Desktop/DisasterTweet/randomforest.pkl'

with open(random_model_path, 'rb') as model_file:
    random_model, random_vectorizer = pickle.load(model_file) 


xgboost_model_path = 'C:/Users/hp/Desktop/DisasterTweet/xgboost.pkl'

with open(xgboost_model_path, 'rb') as model_file:
    xgboost_model, xgboost_vectorizer = pickle.load(model_file) 
    
cat_model_path = 'C:/Users/hp/Desktop/DisasterTweet/catboost.pkl'

with open(cat_model_path, 'rb') as model_file:
    cat_model, cat_vectorizer = pickle.load(model_file)  
    

lightgbm_model_path = 'C:/Users/hp/Desktop/DisasterTweet/lightgbmm.pkl'

with open(lightgbm_model_path, 'rb') as model_file:
    lightgbm_model, lightgbm_vectorizer = pickle.load(model_file) 
    
    
    
stacking_model_path = 'C:/Users/hp/Desktop/DisasterTweet/stacking.pkl'

with open(stacking_model_path, 'rb') as model_file:
    stacking_model, stacking_vectorizer = pickle.load(model_file) 
    
from transformers import AutoModel, BertTokenizerFast
import torch
import torch.nn as nn

from transformers import AutoModel
import torch.nn as nn

class CustomBertModel(nn.Module):
    def __init__(self, bert):
        super(CustomBertModel, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768, 512)  # Additional layer
        self.fc2 = nn.Linear(512, 2)    # Additional layer (output layer)

    def forward(self, sent_id, mask):
        # Get the output from BERT
        outputs = self.bert(sent_id, attention_mask=mask)
        cls_hs = outputs.last_hidden_state[:, 0, :]  # Extract the [CLS] token's hidden state
        # Pass through additional layers
        x = self.fc1(cls_hs)
        x = self.fc2(x)
        return x





# Load the BERT model and tokenizer
bert_model_path = 'C:/Users/hp/Desktop/DisasterTweet/bert_model.pkl'
bert_tokenizer_path = 'C:/Users/hp/Desktop/DisasterTweet/bert_tokenizer.pkl'

# Load the BERT model
bert = AutoModel.from_pretrained('bert-base-uncased')

# Initialize the custom model
model = CustomBertModel(bert)

# Load the state dictionary
state_dict = torch.load(bert_model_path, weights_only=True)

# Load the state dictionary into the custom model
model.load_state_dict(state_dict)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Load the BERT tokenizer
with open(bert_tokenizer_path, 'rb') as tokenizer_file:
    bert_tokenizer = pickle.load(tokenizer_file)


# Download stopwords
nltk.download('stopwords')

class TweetPreprocessor:
    def __init__(self):
        self.known_abbreviations = {
            # Basic Internet Slang
            "u": "you", "pls": "please", "bc": "because", "idk": "i do not know", "omg": "oh my god", 
            "btw": "by the way", "fyi": "for your information", "ttyl": "talk to you later", "lmao": "laughing my ass off",
            # ... (add other abbreviations here)
        }
        self.numeric_abbreviations = {
            "2": "to", "4": "for"
        }

        self.spell = SpellChecker(language='en')
        self.stop_words = set(stopwords.words('english'))

    def _reduce_lengthening(self, text: str) -> str:
        pattern = re.compile(r'(.)\1{2,}')
        return pattern.sub(r'\1\1', text)

    def _tokenize(self, text: str) -> list:
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
        for num in self.numeric_abbreviations:
            text = re.sub(r'\b' + num + r'\b', f'NUM_ABBREV_{num}', text)

        text = re.sub(r'\d+%', 'PERCENTAGE', text)
        text = re.sub(r'\d+k\b', 'THOUSAND', text)
        text = re.sub(r'\$\d+', 'MONEY', text)
        text = re.sub(r'\b\d+\b', 'NUMBER', text)

        for num in self.numeric_abbreviations:
            text = text.replace(f'NUM_ABBREV_{num}', num)

        return text

    def preprocess(self, text: Optional[str]) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower()

        # Emoji processing
        text = self.extract_and_classify_emojis(text)

        # Remove URLs and mentions
        text = re.sub(r'http\S+|www\S+', 'URL', text)
        text = re.sub(r'@\w+', 'USER', text)
        text = re.sub(r'#(\w+)', r'\1', text)

        # Handle numbers and abbreviations
        text = self._normalize_numbers(text)

        # Reduce character lengthening (like "sooome" to "some")
        text = self._reduce_lengthening(text)

        # Tokenization and abbreviation handling
        words = self._tokenize(text)
        words = [self._handle_abbreviation(word) for word in words]

        # Remove stop words
        words = [word for word in words if word not in self.stop_words]

        # Join the words back into a string
        text = ' '.join(words)
        text = re.sub(r'\s+', ' ', text)

        # Correct spelling
        text = self.correct_spelling(text)

        return text.strip()

    def extract_and_classify_emojis(self, text):
        emoji_patterns = {
            r'(?:^|\s)[:;=8][-]?[)}\]D](?:\s|$)': 'positive',
            r'(?:^|\s)[:;=8][-]?[({\[](?:\s|$)': 'negative',
            r'(?:^|\s)XD|xD(?:\s|$)': 'laughing',
            r'(?:^|\s)[:;=8][-]?[Pp](?:\s|$)': 'playful',
            r'(?:^|\s)[:;=8][-]?[Oo](?:\s|$)': 'surprised',
            r'(?:^|\s)[:;=8][-]?[/\\](?:\s|$)': 'skeptical',
            r'(?:^|\s)<3(?:\s|$)': 'love',
            r'(?:^|\s)>3(?:\s|$)': 'broken_heart',
            r'(?:^|\s):[\\|/](?:\s|$)': 'skeptical',
            r'(?:^|\s)=[\\|/](?:\s|$)': 'skeptical'
        }

        if not isinstance(text, str):
            return text

        text = text.replace('=(', ':(')
        text = text.replace('=)', ':)')

        found_emojis = []
        sentiments = []

        for pattern, sentiment in emoji_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                cleaned_matches = [match.strip() for match in matches]
                found_emojis.extend(cleaned_matches)
                sentiments.extend([sentiment] * len(matches))

        if found_emojis:
            text = text + " " + ' '.join(found_emojis)

        return text

    def correct_spelling(self, text):
        words = text.split()
        corrected_words = []
        for word in words:
            if word not in self.spell:
                corrected_word = self.spell.correction(word)
                if corrected_word is not None:
                    corrected_words.append(corrected_word)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words)


def preprocess_tweet(tweet):
    preprocessor = TweetPreprocessor()
    processed_text = preprocessor.preprocess(tweet)
    return processed_text


def tweet_classification(request):
    return render(request, 'tweet_classification.html')

import logging

logger = logging.getLogger(__name__)
def classify_tweet(request):
    if request.method == 'POST':
        try:
            # Parse the JSON data from the request
            logger.info(f"Request data: {request.body}")
            data = json.loads(request.body)
            tweet = data.get('tweet', '')
            model_choice = data.get('model_choice', 'option1')
            logger.info(f"Tweet: {tweet}, Model Choice: {model_choice}")
            # Preprocess the tweet
            processed_text = preprocess_tweet(tweet)
            logger.info(f"Processed Text: {processed_text}")
            # Classify the tweet based on the selected model
            if model_choice == 'option1':
                # Logistic Regression
                tweet_vectorized = log_reg_vectorizer.transform([processed_text])
                prediction = log_reg_model.predict(tweet_vectorized)
                result = int(prediction[0])  # Convert numpy.int64 to int
            elif model_choice == 'option2':
                # SVM
                tweet_vectorized = svm_vectorizer.transform([processed_text])
                prediction = svm_model.predict(tweet_vectorized)
                result = int(prediction[0])  
            elif model_choice == 'option3':
                # random
                tweet_vectorized = random_vectorizer.transform([processed_text])
                prediction = random_model.predict(tweet_vectorized)
                result = int(prediction[0]) 
            elif model_choice == 'option4':
                # xgboost
                tweet_vectorized = xgboost_vectorizer.transform([processed_text])
                prediction = xgboost_model.predict(tweet_vectorized)
                result = int(prediction[0])
            elif model_choice == 'option5':
                # lightgbm
                tweet_vectorized = lightgbm_vectorizer.transform([processed_text])
                prediction = lightgbm_model.predict(tweet_vectorized)
                result = int(prediction[0])
            elif model_choice == 'option6':
                # BERT
                # Tokenize the tweet
                tokens = bert_tokenizer.batch_encode_plus(
                    [processed_text],
                    max_length=25,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
                # Move tensors to the appropriate device
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)
                # Make prediction
                with torch.no_grad():
                    preds = model(input_ids, attention_mask)  # Use the custom model
                    preds = torch.softmax(preds, dim=1)  # Apply softmax to get probabilities
                    preds = preds.argmax(dim=1).cpu().numpy()  # Get the predicted class
                result = int(preds[0])
            elif model_choice == 'option7':
                # lightgbm
                tweet_vectorized = cat_vectorizer.transform([processed_text])
                prediction = cat_model.predict(tweet_vectorized)
                result = int(prediction[0])
            elif model_choice == 'option8':
                # lightgbm
                tweet_vectorized = stacking_vectorizer.transform([processed_text])
                prediction = stacking_model.predict(tweet_vectorized)
                result = int(prediction[0])
            else:
                return JsonResponse({'error': 'Invalid model choice'}, status=400)
            logger.info(f"Prediction Result: {result}")
            # Return the result
            return JsonResponse({'result': result})

        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}", exc_info=True)
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)




