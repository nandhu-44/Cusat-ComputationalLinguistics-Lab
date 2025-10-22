import re
from collections import defaultdict, Counter


def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def compute_translation_probabilities(parallel_corpus):
    """Compute P(f|e) and P(e|f) using IBM Model 1 approach"""
    
    # Count co-occurrences
    ef_counts = defaultdict(Counter)  # e->f counts
    fe_counts = defaultdict(Counter)  # f->e counts
    e_counts = Counter()
    f_counts = Counter()
    
    for english, malayalam in parallel_corpus:
        e_words = tokenize(english)
        f_words = tokenize(malayalam)
        
        for e_word in e_words:
            e_counts[e_word] += 1
            for f_word in f_words:
                ef_counts[e_word][f_word] += 1
        
        for f_word in f_words:
            f_counts[f_word] += 1
            for e_word in e_words:
                fe_counts[f_word][e_word] += 1
    
    # Calculate P(f|e)
    p_f_given_e = defaultdict(dict)
    for e_word in ef_counts:
        total = sum(ef_counts[e_word].values())
        for f_word, count in ef_counts[e_word].items():
            p_f_given_e[e_word][f_word] = count / total
    
    # Calculate P(e|f)
    p_e_given_f = defaultdict(dict)
    for f_word in fe_counts:
        total = sum(fe_counts[f_word].values())
        for e_word, count in fe_counts[f_word].items():
            p_e_given_f[f_word][e_word] = count / total
    
    return p_f_given_e, p_e_given_f


def main():
    parallel_corpus = [
        ("I love reading books", "ഞാൻ പുസ്തകങ്ങൾ വായിക്കാൻ ഇഷ്ടപ്പെടുന്നു"),
        ("She is a good teacher", "അവൾ ഒരു നല്ല അധ്യാപികയാണ്"),
        ("The cat is sleeping", "പൂച്ച ഉറങ്ങുകയാണ്"),
        ("He likes playing cricket", "അവൻ ക്രിക്കറ്റ് കളിക്കാൻ ഇഷ്ടപ്പെടുന്നു"),
        ("We are learning Malayalam", "ഞങ്ങൾ മലയാളം പഠിക്കുന്നു")
    ]
    
    print("=" * 80)
    print("Translation Probability Computation")
    print("=" * 80)
    
    print("\nParallel Corpus:")
    for i, (eng, mal) in enumerate(parallel_corpus, 1):
        print(f"{i}. EN: {eng}")
        print(f"   ML: {mal}\n")
    
    p_f_given_e, p_e_given_f = compute_translation_probabilities(parallel_corpus)
    
    print("=" * 80)
    print("P(Malayalam|English) - Top translations")
    print("=" * 80)
    
    for e_word in sorted(p_f_given_e.keys())[:10]:
        print(f"\n'{e_word}' →")
        sorted_translations = sorted(p_f_given_e[e_word].items(), key=lambda x: x[1], reverse=True)[:3]
        for f_word, prob in sorted_translations:
            print(f"  {f_word}: {prob:.3f}")
    
    print("\n" + "=" * 80)
    print("P(English|Malayalam) - Top translations")
    print("=" * 80)
    
    malayalam_words = sorted(p_e_given_f.keys())[:10]
    for f_word in malayalam_words:
        print(f"\n'{f_word}' →")
        sorted_translations = sorted(p_e_given_f[f_word].items(), key=lambda x: x[1], reverse=True)[:3]
        for e_word, prob in sorted_translations:
            print(f"  {e_word}: {prob:.3f}")
    
    print("\n" + "=" * 80)
    print("Translation Examples")
    print("=" * 80)
    
    test_words = {
        'english': ['love', 'teacher', 'cat', 'learning'],
        'malayalam': ['പുസ്തകങ്ങൾ', 'ഉറങ്ങുകയാണ്', 'പഠിക്കുന്നു']
    }
    
    print("\nEnglish → Malayalam:")
    for e_word in test_words['english']:
        if e_word in p_f_given_e:
            best = max(p_f_given_e[e_word].items(), key=lambda x: x[1])
            print(f"  {e_word} → {best[0]} (P={best[1]:.3f})")
    
    print("\nMalayalam → English:")
    for f_word in test_words['malayalam']:
        if f_word in p_e_given_f:
            best = max(p_e_given_f[f_word].items(), key=lambda x: x[1])
            print(f"  {f_word} → {best[0]} (P={best[1]:.3f})")


if __name__ == "__main__":
    main()
