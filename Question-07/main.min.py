from collections import defaultdict

parallel_corpus = [
    ("the cat is sleeping", "poocha urangunnund"),
    ("the dog is barking", "naay kurakkunnund"),
    ("cat and dog are friends", "poocha naay snehithamar"),
    ("the house is big", "veedu valuthaan"),
    ("big cat in the house", "valiya poocha veetil")
]

def compute_translation_probabilities(corpus):
    """
    Compute P(f|e) and P(e|f) from parallel corpus.
    f = foreign (Malayalam), e = English
    """
    count_ef = defaultdict(lambda: defaultdict(int))
    count_e = defaultdict(int)
    count_f = defaultdict(int)
    
    for english, malayalam in corpus:
        e_words = english.split()
        f_words = malayalam.split()
        
        for e in e_words:
            count_e[e] += 1
            for f in f_words:
                count_ef[e][f] += 1
        
        for f in f_words:
            count_f[f] += 1
    
    p_f_given_e = defaultdict(dict)
    for e in count_ef:
        for f in count_ef[e]:
            p_f_given_e[e][f] = count_ef[e][f] / count_e[e]
    
    p_e_given_f = defaultdict(dict)
    for e in count_ef:
        for f in count_ef[e]:
            p_e_given_f[f][e] = count_ef[e][f] / count_f[f]
    
    return p_f_given_e, p_e_given_f, count_ef, count_e, count_f


def display_results(p_f_given_e, p_e_given_f, count_ef, count_e, count_f):
    """Display translation probabilities in a readable format."""
    print("PARALLEL CORPUS:")
    for i, (eng, mal) in enumerate(parallel_corpus, 1):
        print(f"{i}. English:   {eng}")
        print(f"   Malayalam: {mal}")
    
    print("\nWORD COUNTS:")
    print("English words:")
    for word, count in sorted(count_e.items()):
        print(f"  {word}: {count}")
    
    print("\nMalayalam words:")
    for word, count in sorted(count_f.items()):
        print(f"  {word}: {count}")
    
    print("\nTRANSLATION PROBABILITIES P(Malayalam|English):")
    for e in sorted(p_f_given_e.keys()):
        print(f"'{e}' translates to:")
        for f, prob in sorted(p_f_given_e[e].items(), key=lambda x: -x[1]):
            print(f"  '{f}': {prob:.3f} ({count_ef[e][f]}/{count_e[e]})")
    
    print("\nTRANSLATION PROBABILITIES P(English|Malayalam):")
    for f in sorted(p_e_given_f.keys()):
        print(f"'{f}' translates to:")
        for e, prob in sorted(p_e_given_f[f].items(), key=lambda x: -x[1]):
            print(f"  '{e}': {prob:.3f} ({count_ef[e][f]}/{count_f[f]})")


def translate_word(word, p_trans, direction):
    """Get most likely translation for a word."""
    if word not in p_trans:
        return None
    translations = p_trans[word]
    best_translation = max(translations.items(), key=lambda x: x[1])
    print(f"Most likely translation of '{word}' ({direction}):")
    print(f"  '{best_translation[0]}' with probability {best_translation[1]:.3f}")
    return best_translation

if __name__ == "__main__":
    p_f_given_e, p_e_given_f, count_ef, count_e, count_f = compute_translation_probabilities(parallel_corpus)
    
    display_results(p_f_given_e, p_e_given_f, count_ef, count_e, count_f)
    
    print("\nEXAMPLE TRANSLATIONS:")
    
    translate_word("cat", p_f_given_e, "English → Malayalam")
    translate_word("poocha", p_e_given_f, "Malayalam → English")
    translate_word("house", p_f_given_e, "English → Malayalam")
    translate_word("veedu", p_e_given_f, "Malayalam → English")