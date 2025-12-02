from collections import Counter

dictionary = {
    "the": 1000, "word": 500, "world": 450, "work": 400,
    "wood": 300, "bird": 200, "cord": 150, "lord": 120
}
V = list(dictionary.keys())
word_freq = Counter(dictionary)

misspelled = "wrod"
candidates = ["word", "wood", "world"]

print(f"Misspelled word: '{misspelled}'")
print(f"Candidates     : {candidates}\n")
print("-" * 60)

P_error = 0.05

total_words = sum(word_freq.values())
P_w = {w: word_freq[w] / total_words for w in V}

scores = {}
for w in candidates:
    P_s_given_w = P_error
    score = P_s_given_w * P_w[w]
    scores[w] = score
    print(f"P(s|{w:5}) = {P_s_given_w:.3f} × P({w}) = {P_w[w]:.5f} → Score = {score:.7f}")

best_word = max(scores, key=scores.get)

print("-" * 60)
print(f"Best correction → '{best_word}' with score {scores[best_word]:.7f}")
print(f"Original intent → 'word' {'Correct' if best_word == 'word' else 'Incorrect'}")