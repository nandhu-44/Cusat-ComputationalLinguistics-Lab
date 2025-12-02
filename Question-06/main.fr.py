import stanza
from tabulate import tabulate

stanza.download('fr', verbose=True)
nlp_fr = stanza.Pipeline('fr', processors='tokenize,pos', verbose=False)

nlp_en = stanza.Pipeline('en', processors='tokenize,pos', verbose=False)

french_sentences = [
    "Le chat mange une souris.",
    "Les enfants jouent dans le jardin.",
    "Marie et Paul sont très heureux aujourd'hui.",
    "Je vais à l'école en autobus.",
    "Nous aimons manger du fromage et boire du vin."
]

print("FRENCH POS TAGGING RESULTS\n" + "="*70)
fr_results = []

for sent in french_sentences:
    doc = nlp_fr(sent)
    print(f"\nSentence: {sent}")
    row = []
    for token in doc.sentences[0].words:
        print(f"{token.text:12} → {token.upos}")
        row.append([token.text, token.upos, token.feats if token.feats else "-"])
    fr_results.append(row)
    print("-" * 50)

all_fr_tags = [w.upos for s in nlp_fr(" ".join(french_sentences)).sentences for w in s.words]
tag_counts_fr = {}
for tag in all_fr_tags:
    tag_counts_fr[tag] = tag_counts_fr.get(tag, 0) + 1

print("\nCOMMON FRENCH POS TAGS IN DATASET:")
print(tabulate(sorted(tag_counts_fr.items(), key=lambda x: -x[1]), 
               headers=["Tag", "Count"], tablefmt="github"))

print("\n\nENGLISH EQUIVALENT (for reference):")
en_example = nlp_en("The cat eats a mouse.")
for w in en_example.sentences[0].words:
    print(f"{w.text:10} → {w.upos}")

with open("french_pos_tagging.txt", "w", encoding="utf-8") as f:
    f.write("French POS Tagging Output\n" + "="*50 + "\n")
    for sent in french_sentences:
        f.write(f"\n{sent}\n")
        doc = nlp_fr(sent)
        for w in doc.sentences[0].words:
            f.write(f"{w.text:15} {w.upos}  [{w.feats}]\n")