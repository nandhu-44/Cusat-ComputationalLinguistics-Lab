import stanza
import sys

print("=" * 80)
print("PART 1: Load sentences in Hindi")
print("=" * 80)

sentences = [
    "मैं स्कूल जाता हूँ।",
    "वह किताब पढ़ रही है।",
    "राम और श्याम दोस्त हैं।",
    "लड़की ने आम खाया।"
]

for i, sent in enumerate(sentences, 1):
    print(f"{i}. {sent}")

print("\n" + "=" * 80)
print("PART 2: POS Tagging using Stanza")
print("=" * 80)
print("Downloading Hindi model...")

try:
    stanza.download('hi', verbose=False)
    nlp = stanza.Pipeline('hi', processors='tokenize,pos', verbose=False, use_gpu=False)
    
    for sent in sentences:
        doc = nlp(sent)
        print(f"\nSentence: {sent}")
        for sentence in doc.sentences:
            for word in sentence.words:
                print(f"  {word.text:15} UPOS: {word.upos:10} XPOS: {word.xpos}")
    
    print("\n" + "=" * 80)
    print("PART 3: Common Tag Types - Hindi vs English")
    print("=" * 80)
    
    all_tags = {}
    for sent in sentences:
        doc = nlp(sent)
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos not in all_tags:
                    all_tags[word.upos] = []
                all_tags[word.upos].append(word.text)
    
    tag_names = {
        "PRON": "Pronoun",
        "NOUN": "Noun",
        "VERB": "Verb",
        "ADP": "Adposition",
        "ADJ": "Adjective",
        "CCONJ": "Coordinating Conjunction",
        "AUX": "Auxiliary",
        "PROPN": "Proper Noun",
        "PUNCT": "Punctuation",
        "DET": "Determiner",
        "NUM": "Numeral"
    }
    
    print(f"{'Tag':<10} {'Type':<30} {'Examples from Text':<30}")
    print("-" * 80)
    for tag, examples in sorted(all_tags.items()):
        tag_name = tag_names.get(tag, "Other")
        examples_str = ", ".join(examples[:3])
        print(f"{tag:<10} {tag_name:<30} {examples_str:<30}")

except Exception as e:
    print(f"Error with Stanza: {e}")
    print("Network or model download failed. Install stanza and run with stable internet.")
    sys.exit(1)
