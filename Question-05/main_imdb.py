"""Naive Bayes classifier for IMDB 50K movie reviews dataset"""

import re
import csv
from collections import defaultdict, Counter
from math import log
from random import sample


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'


def colorize(text, sentiment):
    return f"{Colors.GREEN}{text}{Colors.RESET}" if sentiment == 'positive' else f"{Colors.RED}{text}{Colors.RESET}"


class NaiveBayesSentimentClassifier:
    def __init__(self):
        self.class_counts = Counter()
        self.word_counts = defaultdict(Counter)
        self.vocabulary = set()
        self.total_docs = 0

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.split()

    def train(self, documents, labels):
        self.total_docs = len(documents)
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            for word in self.preprocess(doc):
                self.vocabulary.add(word)
                self.word_counts[label][word] += 1

    def predict(self, document):
        class_probs = {}
        for class_label in self.class_counts:
            words = self.preprocess(document)
            log_prob = log(self.class_counts[class_label] / self.total_docs)
            total_words = sum(self.word_counts[class_label].values())
            vocab_size = len(self.vocabulary)
            
            for word in words:
                count = self.word_counts[class_label][word]
                log_prob += log((count + 1) / (total_words + vocab_size))
            
            class_probs[class_label] = log_prob
        
        return max(class_probs, key=class_probs.get)

    def evaluate(self, test_docs, test_labels):
        correct = sum(1 for doc, true in zip(test_docs, test_labels) if self.predict(doc) == true)
        return correct / len(test_docs)


def load_imdb_data(filename='IMDB Dataset.csv', train_size=5000, test_size=1000):
    """Load IMDB dataset and split into train/test"""
    reviews, sentiments = [], []
    
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'review' in row and 'sentiment' in row:
                reviews.append(row['review'])
                sentiments.append(row['sentiment'].strip().lower())
    
    train_docs = reviews[:train_size]
    train_labels = sentiments[:train_size]
    test_docs = reviews[train_size:train_size + test_size]
    test_labels = sentiments[train_size:train_size + test_size]
    
    return train_docs, train_labels, test_docs, test_labels


def main():
    print("=" * 80)
    print("IMDB 50K Dataset - Naive Bayes Classifier")
    print("=" * 80)
    
    train_docs, train_labels, test_docs, test_labels = load_imdb_data(
        filename='IMDB Dataset.csv',
        train_size=5000,
        test_size=1000
    )
    
    print(f"\nTrain: {len(train_docs)} | Test: {len(test_docs)}")
    
    train_counts = Counter(train_labels)
    for sentiment, count in train_counts.items():
        print(f"  {colorize(sentiment, sentiment)}: {count}")
    
    print(f"\n{Colors.CYAN}Training...{Colors.RESET}")
    classifier = NaiveBayesSentimentClassifier()
    classifier.train(train_docs, train_labels)
    print(f"Vocabulary: {len(classifier.vocabulary)} words")
    
    print("\n" + "=" * 80)
    print("Sample Predictions (20 random)")
    print("=" * 80)
    
    random_indices = sample(range(len(test_docs)), min(20, len(test_docs)))
    
    for idx in random_indices:
        doc = test_docs[idx]
        true = test_labels[idx]
        pred = classifier.predict(doc)
        
        text = doc[:60] + '...' if len(doc) > 60 else doc
        symbol = '✓' if pred == true else '✗'
        color = Colors.GREEN if pred == true else Colors.RED
        print(f"{color}{symbol}{Colors.RESET} \"{text}\"")
        print(f"  True: {colorize(true, true)} → Pred: {colorize(pred, pred)}\n")
    
    print("=" * 80)
    print("Final Results")
    print("=" * 80)
    
    accuracy = classifier.evaluate(test_docs, test_labels)
    predictions = [classifier.predict(doc) for doc in test_docs]
    correct = sum(1 for p, t in zip(predictions, test_labels) if p == t)
    
    acc_color = Colors.GREEN if accuracy > 0.7 else Colors.RED
    print(f"Accuracy: {acc_color}{accuracy * 100:.2f}%{Colors.RESET} ({correct}/{len(test_labels)})")
    
    tp = sum(1 for p, t in zip(predictions, test_labels) if p == 'positive' and t == 'positive')
    tn = sum(1 for p, t in zip(predictions, test_labels) if p == 'negative' and t == 'negative')
    fp = sum(1 for p, t in zip(predictions, test_labels) if p == 'positive' and t == 'negative')
    fn = sum(1 for p, t in zip(predictions, test_labels) if p == 'negative' and t == 'positive')
    
    print(f"TP: {Colors.GREEN}{tp}{Colors.RESET} | TN: {Colors.GREEN}{tn}{Colors.RESET} | FP: {Colors.RED}{fp}{Colors.RESET} | FN: {Colors.RED}{fn}{Colors.RESET}")


if __name__ == "__main__":
    main()
