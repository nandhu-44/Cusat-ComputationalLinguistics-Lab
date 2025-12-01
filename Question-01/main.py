import nltk
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def process_text_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        paragraph = file.read()
    
    tokens = nltk.word_tokenize(paragraph)
    
    tokens = [word.lower() for word in tokens if word not in string.punctuation]
    
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    stemmed = [stemmer.stem(word) for word in tokens]
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    
    output = f"""File: {filename}\n\n
Original Tokens : {tokens}\n\n
Stemmed         : {stemmed}\n\n
Lemmatized      : {lemmatized}"""
    print(output)
    with open("output.txt", "w", encoding='utf-8') as out_file:
        out_file.write(output)

process_text_from_file("./paragraph.txt")