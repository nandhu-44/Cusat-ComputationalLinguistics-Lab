# Q7 - FINAL CORRECT VERSION: IBM Model 1 (English → French)
from collections import defaultdict

# === 5 Clean Parallel Sentences ===
english = [
    "the cat is on the mat",
    "a dog runs in the park",
    "i love this beautiful book",
    "we are very happy today",
    "she reads a newspaper"
]

french = [
    "le chat est sur le tapis",
    "un chien court dans le parc",
    "j aime ce beau livre",
    "nous sommes très heureux aujourd hui",
    "elle lit un journal"
]

# Tokenize
e_sents = [sent.split() for sent in english]
f_sents = [sent.split() for sent in french]

# Vocabularies
e_vocab = set(word for sent in e_sents for word in sent)
f_vocab = set(word for sent in f_sents for word in sent)

# === ADD NULL TOKEN PROPERLY ===
NULL = None  # We'll handle it specially
e_sents_with_null = [["NULL"] + sent for sent in e_sents]  # Prepend NULL to every English sentence
e_vocab.add("NULL")

# Initialize t(f|e) uniformly
t = defaultdict(lambda: defaultdict(lambda: 1.0 / len(e_vocab)))

print("Running IBM Model 1 (10 iterations)...\n")

# === EM Algorithm (10 iterations) ===
for iteration in range(10):
    count = defaultdict(lambda: defaultdict(float))
    total = defaultdict(float)

    for e_sent, f_sent in zip(e_sents_with_null, f_sents):
        # Compute normalization
        for f_word in f_sent:
            denom = sum(t[f_word][e_word] for e_word in e_sent)
            for e_word in e_sent:
                delta = t[f_word][e_word] / denom
                count[f_word][e_word] += delta
                total[e_word] += delta

    # Update t(f|e)
    for f_word in f_vocab:
        for e_word in list(e_vocab):
            if total[e_word] > 0:
                t[f_word][e_word] = count[f_word][e_word] / total[e_word]
            else:
                t[f_word][e_word] = 0.0

    if iteration + 1 in [1, 5, 10]:
        print(f"Iteration {iteration+1:2d}: P(chat|cat) = {t['chat']['cat']:.4f}, "
              f"P(le|the) = {t['le']['the']:.4f}, P(livre|book) = {t['livre']['book']:.4f}")

# === FINAL RESULTS ===
print("\n" + "="*60)
print("TOP TRANSLATIONS P(f|e) AFTER 10 ITERATIONS")
print("="*60)
results = []
for e in sorted(e_vocab - {"NULL"}):
    if e not in ["NULL"]:
        best_f = max(f_vocab, key=lambda f: t[f][e])
        prob = t[best_f][e]
        results.append((e, best_f, round(prob, 4)))

# Sort by English word
for e, f, p in sorted(results):
    print(f"{e:12} → {f:12} ({p})")