"""Implement a text classifier for sentiment analysis using naive bayes theorem"""

import re
from collections import defaultdict, Counter
from math import log


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
    
    train_data = [
        ("I love this movie it is amazing", "positive"),
        ("This is a great film wonderful acting", "positive"),
        ("Excellent movie highly recommended", "positive"),
        ("Best film I have seen in years", "positive"),
        ("Fantastic performance by the actors", "positive"),
        ("I hate this movie it is terrible", "negative"),
        ("This film is awful and boring", "negative"),
        ("Worst movie ever made", "negative"),
        ("Terrible acting and bad plot", "negative"),
        ("Horrible film waste of time", "negative"),
    ]
    
    test_data = [
        ("This movie is great and amazing", "positive"),
        ("I really love the wonderful acting", "positive"),
        ("This is terrible and awful", "negative"),
        ("Horrible movie I hate it", "negative"),
    ]
    
    train_docs = [doc for doc, _ in train_data]
    train_labels = [label for _, label in train_data]
    test_docs = [doc for doc, _ in test_data]
    test_labels = [label for _, label in test_data]
    
    print("\nTraining Data:")
    print("-" * 80)
    for doc, label in train_data[:3]:
        colored_label = colorize(label.upper(), label)
        print(f"[{colored_label}] {doc}")
    print(f"... ({len(train_data)} total training samples)")
    
    classifier = NaiveBayesSentimentClassifier()
    classifier.train(train_docs, train_labels)
    
    print("\n" + "=" * 80)
    print("Model Statistics")
    print("=" * 80)
    print(f"Vocabulary size: {len(classifier.vocabulary)}")
    print(f"Class distribution:")
    for label, count in classifier.class_counts.items():
        prob = count / classifier.total_docs
        print(f"  {label}: {count} documents (P({label}) = {prob:.3f})")
    
    print("\n" + "=" * 80)
    print("Testing Phase")
    print("=" * 80)
    
    for doc, true_label in test_data:
        pred_label, probs = classifier.predict(doc)
        print(f"\nDocument: \"{doc}\"")
        print(f"True label: {colorize(true_label, true_label)}")
        print(f"Predicted: {colorize(pred_label, pred_label)}")
        print(f"Log probabilities:")
        for label, prob in probs.items():
            colored_label = colorize(label, label)
            print(f"  P({colored_label}|document) ∝ {prob:.4f}")
        result_color = Colors.GREEN if pred_label == true_label else Colors.RED
        print(f"Result: {result_color}{'✓ CORRECT' if pred_label == true_label else '✗ WRONG'}{Colors.RESET}")
    
    accuracy, predictions = classifier.evaluate(test_docs, test_labels)
    
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Correct predictions: {sum(p == t for p, t in zip(predictions, test_labels))}/{len(test_labels)}")
    
    print("\n" + "=" * 80)
    print("Interactive Testing")
    print("=" * 80)
    
    custom_tests = [
        "This is absolutely wonderful",
        "I hate everything about this",
        "Amazing and fantastic experience",
        "Boring and terrible waste"
    ]
    
    for test_text in custom_tests:
        pred, probs = classifier.predict(test_text)
        print(f"\n\"{test_text}\"")
        print(f"→ Predicted: {colorize(pred.upper(), pred)}")


if __name__ == "__main__":
    main()
