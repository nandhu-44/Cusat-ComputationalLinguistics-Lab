import stanza
import sys
from collections import defaultdict, Counter

print("=" * 80)
print("PART 1: Load sentences in Hindi")
print("=" * 80)

sentences = [
    "मैं स्कूल जाता हूँ।",
    "वह किताब पढ़ रही है।",
    "राम और श्याम दोस्त हैं।",
    "लड़की ने आम खाया।",
    "बच्चे खेल रहे थे।",
    "यह बहुत सुंदर फूल है।",
    "उसने मुझे एक पत्र लिखा।"
]

for i, sent in enumerate(sentences, 1):
    print(f"{i}. {sent}")

print("\n" + "=" * 80)
print("PART 2: POS Tagging using Stanza with Morphological Features")
print("=" * 80)
print("Loading Hindi model...")

try:
    stanza.download('hi', verbose=False)
    nlp = stanza.Pipeline('hi', processors='tokenize,pos,lemma', verbose=False, use_gpu=False)
    
    tagged_docs = []
    for sent in sentences:
        doc = nlp(sent)
        tagged_docs.append(doc)
        print(f"\nSentence: {sent}")
        print(f"{'Word':<15} {'Lemma':<15} {'UPOS':<10} {'XPOS':<8} {'Morphological Features'}")
        print("-" * 80)
        for sentence in doc.sentences:
            for word in sentence.words:
                feats = word.feats if word.feats else "None"
                print(f"{word.text:<15} {word.lemma:<15} {word.upos:<10} {word.xpos:<8} {feats}")
    
    print("\n" + "=" * 80)
    print("PART 3: Statistical Analysis & Comparison with English")
    print("=" * 80)
    
    tag_counter = Counter()
    tag_examples = defaultdict(set)
    morphological_complexity = defaultdict(list)
    
    for doc in tagged_docs:
        for sentence in doc.sentences:
            for word in sentence.words:
                tag_counter[word.upos] += 1
                tag_examples[word.upos].add(word.text)
                if word.feats:
                    morphological_complexity[word.upos].append(word.feats)
    
    tag_names = {
        "PRON": ("Pronoun", "I, he, she"),
        "NOUN": ("Noun", "school, book, mango"),
        "VERB": ("Verb", "go, read, write"),
        "ADP": ("Adposition/Postposition", "to, from (prepositions in English)"),
        "ADJ": ("Adjective", "good, beautiful"),
        "CCONJ": ("Coordinating Conjunction", "and, but"),
        "AUX": ("Auxiliary", "is, are, was"),
        "PROPN": ("Proper Noun", "Ram, Shyam"),
        "PUNCT": ("Punctuation", "., !, ?"),
        "DET": ("Determiner", "the, a, this"),
        "NUM": ("Numeral", "one, two"),
        "ADV": ("Adverb", "very, quickly"),
        "PART": ("Particle", "not, to")
    }
    
    print(f"\n{'Tag':<10} {'Count':<8} {'Type':<30} {'Hindi Examples':<30} {'English Equivalent':<25}")
    print("=" * 120)
    for tag, count in tag_counter.most_common():
        tag_info = tag_names.get(tag, ("Other", "N/A"))
        examples_str = ", ".join(list(tag_examples[tag])[:4])
        print(f"{tag:<10} {count:<8} {tag_info[0]:<30} {examples_str:<30} {tag_info[1]:<25}")
    
    print("\n" + "=" * 80)
    print("Morphological Richness Analysis")
    print("=" * 80)
    
    for tag in ['VERB', 'NOUN', 'AUX']:
        if tag in morphological_complexity and morphological_complexity[tag]:
            unique_features = set(morphological_complexity[tag])
            print(f"\n{tag}: {len(unique_features)} unique morphological patterns")
            for feat in list(unique_features)[:3]:
                print(f"  - {feat}")
    
    print("\n" + "=" * 80)
    print("Dependency Relations (Sample)")
    print("=" * 80)
    
    nlp_full = stanza.Pipeline('hi', processors='tokenize,pos,lemma,depparse', verbose=False, use_gpu=False)
    sample_doc = nlp_full(sentences[0])
    
    print(f"Sentence: {sentences[0]}\n")
    print(f"{'Word':<15} {'Relation':<15} {'Head':<15}")
    print("-" * 50)
    for sentence in sample_doc.sentences:
        for word in sentence.words:
            head_text = sentence.words[word.head - 1].text if word.head > 0 else "ROOT"
            print(f"{word.text:<15} {word.deprel:<15} {head_text:<15}")

except Exception as e:
    print(f"Error with Stanza: {e}")
    print("Network or model download failed. Install stanza and run with stable internet.")
    sys.exit(1)

print("Summary Statistics")
print("=" * 80)
print(f"Total sentences: {len(sentences)}")
print(f"Total tokens: {sum(tag_counter.values())}")
print(f"Unique POS tags: {len(tag_counter)}")
print(f"Most common tag: {tag_counter.most_common(1)[0][0]} ({tag_counter.most_common(1)[0][1]} occurrences)")
