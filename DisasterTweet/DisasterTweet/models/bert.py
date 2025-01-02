import numpy as np

from sklearn.model_selection import train_test_split
from transformers import AutoModel, BertTokenizerFast
from scipy.sparse import hstack
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight



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





"""BERT
In short, BERT is a pre-trained language model that is fine-tuned on specific tasks. 
It processes the text as a whole and understands the meaning of words based on both their left and right context, making it very powerful for natural language processing tasks.
"""
# Diviser les données en ensembles d'entraînement et de test
train_text, temp_text, train_labels, temp_labels = train_test_split(
    data['processed_text'], data['target'], random_state=2018, test_size=0.3, stratify=data['target']
)
# Diviser l'ensemble temporaire en ensembles de validation et de test
val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text, temp_labels, random_state=2018, test_size=0.5, stratify=temp_labels
)               
# Charger le modèle BERT pré-entraîné
bert = AutoModel.from_pretrained('bert-base-uncased')
# Charger le tokenizer BERT
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# Tokenisation et encodage des séquences
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length=25,
    pad_to_max_length=True,
    truncation=True,
    return_tensors="pt"
)

tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length=25,
    pad_to_max_length=True,
    truncation=True,
    return_tensors="pt"
)

tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length=25,
    pad_to_max_length=True,
    truncation=True,
    return_tensors="pt"
)

# Convertir les séquences en tenseurs
train_seq = tokens_train['input_ids']
train_mask = tokens_train['attention_mask']
train_y = torch.tensor(train_labels.values)

val_seq = tokens_val['input_ids']
val_mask = tokens_val['attention_mask']
val_y = torch.tensor(val_labels.values)

test_seq = tokens_test['input_ids']
test_mask = tokens_test['attention_mask']
test_y = torch.tensor(test_labels.values)

batch_size = 32
# Entraînement
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Validation
val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Initialiser le modèle
model = BERT_Arch(bert)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Calcul des poids de classe
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
# Fonction de perte
cross_entropy = nn.NLLLoss(weight=weights)
# Optimiseur
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

def train():
    model.train()
    total_loss = 0
    total_preds = []
    for step, batch in enumerate(train_dataloader):
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        model.zero_grad()
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)
    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

def evaluate():
    model.eval()
    total_loss = 0
    total_preds = []
    for batch in val_dataloader:
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            total_loss += loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
    avg_loss = total_loss / len(val_dataloader)
    return avg_loss

best_valid_loss = float('inf')

for epoch in range(5):
    train_loss = train()
    valid_loss = evaluate()
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    print(f'Epoch {epoch+1}, Training Loss: {train_loss:.3f}, Validation Loss: {valid_loss:.3f}')
model.load_state_dict(torch.load('saved_weights.pt'))
"""# Prédictions
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
"""
# Save the BERT model
torch.save(model.state_dict(), 'bert_model.pkl')
import pickle 
# Save the tokenizer
with open('bert_tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)