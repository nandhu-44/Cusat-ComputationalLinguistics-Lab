import re
import csv
from collections import defaultdict, Counter
from math import log

class NaiveBayesIMDB:
    def __init__(self):
        self.vocab = set()
        self.class_count = Counter()
        self.word_count = defaultdict(Counter)
        self.total_docs = 0

    def preprocess(self, text):
        return re.sub(r'[^a-z\s]', '', text.lower()).split()

    def train(self, docs, labels):
        self.total_docs = len(docs)
        for doc, label in zip(docs, labels):
            self.class_count[label] += 1
            for word in self.preprocess(doc):
                self.vocab.add(word)
                self.word_count[label][word] += 1

    def predict(self, doc):
        words = self.preprocess(doc)
        scores = {}
        for label in self.class_count:
            log_prob = log(self.class_count[label] / self.total_docs)
            total = sum(self.word_count[label].values())
            for word in words:
                count = self.word_count[label][word]
                log_prob += log((count + 1) / (total + len(self.vocab)))
            scores[label] = log_prob
        return max(scores, key=scores.get)

    def evaluate(self, docs, labels):
        correct = sum(1 for doc, true in zip(docs, labels) if self.predict(doc) == true)
        return correct / len(docs) * 100

def load_imdb(n_train=5000, n_test=1000):
    with open('IMDB Dataset.csv', encoding='utf-8', errors='ignore') as f:
        data = list(csv.DictReader(f))
    
    train = data[:n_train]
    test  = data[n_train:n_train + n_test]
    
    return ([r['review'] for r in train], [r['sentiment'] for r in train],
            [r['review'] for r in test],  [r['sentiment'] for r in test])

print("Naive Bayes - IMDB Sentiment Analysis")
train_X, train_y, test_X, test_y = load_imdb(5000, 1000)

print(f"Training on {len(train_X)} | Testing on {len(test_X)} reviews")

nb = NaiveBayesIMDB()
nb.train(train_X, train_y)

print(f"Vocabulary size: {len(nb.vocab)}")
print(f"Accuracy: {nb.evaluate(test_X, test_y):.2f}%")

for text in [
    "This movie was absolutely amazing and brilliant!",
    "Worst film ever, complete waste of time.",
    "I loved it so much, best acting!",
    "Boring, predictable, hated every minute."
]:
    pred = nb.predict(text)
    print(f"\"{text}\" → {pred.upper()}")