import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from spacy import displacy
import re

# Load NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy language model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize text into sentences
    sentences = sent_tokenize(text)

    # Tokenize each sentence into words
    tokens = []
    for sentence in sentences:
        tokens.extend(word_tokenize(sentence))

    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return tokens

text = "Design and evaluate prompts to improve the performance of a given AI model"
preprocessed_text = preprocess_text(text)
print(preprocessed_text)