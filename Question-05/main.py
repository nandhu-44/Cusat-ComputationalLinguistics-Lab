"""Implement a text classifier for sentiment analysis using naive bayes theorem"""

import re
import csv
import os
from collections import defaultdict, Counter
from math import log
from random import sample


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def colorize(text, sentiment):
    if sentiment.lower() == 'positive':
        return f"{Colors.GREEN}{text}{Colors.RESET}"
    elif sentiment.lower() == 'negative':
        return f"{Colors.RED}{text}{Colors.RESET}"
    return text


def load_csv_data(train_file='train.csv', test_file='test.csv', max_samples=None):
    """Load sentiment data from CSV files"""
    train_docs, train_labels = [], []
    test_docs, test_labels = [], []
    
    if os.path.exists(train_file):
        with open(train_file, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'text' in row and 'sentiment' in row:
                    sentiment = row['sentiment'].strip().lower()
                    if sentiment in ['positive', 'negative']:
                        train_docs.append(row['text'])
                        train_labels.append(sentiment)
                        if max_samples and len(train_docs) >= max_samples:
                            break
    
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'text' in row and 'sentiment' in row:
                    sentiment = row['sentiment'].strip().lower()
                    if sentiment in ['positive', 'negative']:
                        test_docs.append(row['text'])
                        test_labels.append(sentiment)
                        if max_samples and len(test_docs) >= max_samples // 4:
                            break
    
    return train_docs, train_labels, test_docs, test_labels


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
            words = self.preprocess(doc)
            
            for word in words:
                self.vocabulary.add(word)
                self.word_counts[label][word] += 1

    def calculate_log_probability(self, document, class_label):
        words = self.preprocess(document)
        
        log_prob = log(self.class_counts[class_label] / self.total_docs)
        
        total_words_in_class = sum(self.word_counts[class_label].values())
        vocab_size = len(self.vocabulary)
        
        for word in words:
            word_count = self.word_counts[class_label][word]
            log_prob += log((word_count + 1) / (total_words_in_class + vocab_size))
        
        return log_prob

    def predict(self, document):
        class_probs = {}
        
        for class_label in self.class_counts:
            class_probs[class_label] = self.calculate_log_probability(document, class_label)
        
        return max(class_probs, key=class_probs.get), class_probs

    def evaluate(self, test_documents, test_labels):
        correct = 0
        predictions = []
        
        for doc, true_label in zip(test_documents, test_labels):
            pred_label, _ = self.predict(doc)
            predictions.append(pred_label)
            if pred_label == true_label:
                correct += 1
        
        accuracy = correct / len(test_documents)
        return accuracy, predictions


def main():
    print("=" * 80)
    print("Naive Bayes Sentiment Analysis Classifier")
    print("=" * 80)
    
    train_docs, train_labels, test_docs, test_labels = load_csv_data(
        train_file='train.csv',
        test_file='test.csv',
        max_samples=2000
    )
    
    print(f"\nTrain: {len(train_docs)} | Test: {len(test_docs)}")
    
    train_sentiment_counts = Counter(train_labels)
    for sentiment, count in train_sentiment_counts.items():
        print(f"  {colorize(sentiment, sentiment)}: {count}")
    
    classifier = NaiveBayesSentimentClassifier()
    classifier.train(train_docs, train_labels)
    
    print(f"\nVocabulary: {len(classifier.vocabulary)} words")
    
    print("\n" + "=" * 80)
    print("Sample Predictions (20 random)")
    print("=" * 80)
    
    random_indices = sample(range(len(test_docs)), min(20, len(test_docs)))
    
    for idx in random_indices:
        doc = test_docs[idx]
        true_label = test_labels[idx]
        pred_label, _ = classifier.predict(doc)
        
        text = doc[:60] + '...' if len(doc) > 60 else doc
        result = '✓' if pred_label == true_label else '✗'
        color = Colors.GREEN if pred_label == true_label else Colors.RED
        print(f"{color}{result}{Colors.RESET} \"{text}\"")
        print(f"  True: {colorize(true_label, true_label)} → Pred: {colorize(pred_label, pred_label)}\n")
    
    print("=" * 80)
    print("Final Results")
    print("=" * 80)
    accuracy, predictions = classifier.evaluate(test_docs, test_labels)
    
    correct = sum(p == t for p, t in zip(predictions, test_labels))
    acc_color = Colors.GREEN if accuracy > 0.7 else Colors.RED
    print(f"Accuracy: {acc_color}{accuracy * 100:.2f}%{Colors.RESET} ({correct}/{len(test_labels)})")
    
    tp = sum(1 for p, t in zip(predictions, test_labels) if p == 'positive' and t == 'positive')
    tn = sum(1 for p, t in zip(predictions, test_labels) if p == 'negative' and t == 'negative')
    fp = sum(1 for p, t in zip(predictions, test_labels) if p == 'positive' and t == 'negative')
    fn = sum(1 for p, t in zip(predictions, test_labels) if p == 'negative' and t == 'positive')
    
    print(f"TP: {Colors.GREEN}{tp}{Colors.RESET} | TN: {Colors.GREEN}{tn}{Colors.RESET} | FP: {Colors.RED}{fp}{Colors.RESET} | FN: {Colors.RED}{fn}{Colors.RESET}")
    
    print("\n" + "=" * 80)
    print("Custom Examples")
    print("=" * 80)
    
    tests = [
        "This is absolutely wonderful and amazing",
        "I hate everything about this terrible movie",
        "Best experience ever, highly recommend",
        "Boring waste of time"
    ]
    
    for text in tests:
        pred, _ = classifier.predict(text)
        print(f"\"{text[:50]}{'...' if len(text) > 50 else ''}\" → {colorize(pred.upper(), pred)}")


if __name__ == "__main__":
    main()
