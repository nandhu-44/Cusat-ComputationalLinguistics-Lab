"""
Noisy Channel Model Spelling Corrector

This script implements a simple noisy channel model to correct a single non-word spelling error.

Dictionary (V): A set of common words.
Misspelled word (s): "wrod" (transposition error from "word").
Candidates: Manually selected plausible corrections from V.
Assumptions:
- P(s|w) = 0.001 for single-edit errors (substitution, insertion, deletion, transposition).
- P(s|w) = 0 for no edit or multiple edits.
- P(w) = uniform prior (1 / |V|) for simplicity.

Formula: Best correction w* = argmax_{w in candidates} P(s|w) * P(w)
"""

from typing import List, Dict, Set
import re


def edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance (edit distance) between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def is_single_edit(s: str, w: str) -> bool:
    """Check if s can be derived from w with exactly one edit."""
    return edit_distance(s, w) == 1


def noisy_channel_corrector(s: str, candidates: List[str], V: Set[str], prior_probs: Dict[str, float]) -> str:
    """
    Correct the misspelled word s using noisy channel model.

    Args:
        s: Misspelled word.
        candidates: List of candidate corrections.
        V: Dictionary of valid words.
        prior_probs: Dictionary of prior probabilities for words in V.

    Returns:
        The best correction based on P(s|w) * P(w).
    """
    # Assumptions
    P_edit = 0.001  # Probability of single-edit error
    P_no_edit = 0.0  # Probability of no error (since s is misspelled)

    best_score = -1
    best_word = None

    for w in candidates:
        if w not in V:
            continue  # Skip if not in dictionary
        if is_single_edit(s, w):
            P_sw = P_edit
        else:
            P_sw = P_no_edit
        P_w = prior_probs.get(w, 0.0)
        score = P_sw * P_w
        if score > best_score:
            best_score = score
            best_word = w

    return best_word if best_word else "No correction found"


def main():
    # Dictionary V
    V = {"word", "world", "work", "worm", "worth"}

    # Prior probabilities for each word in V (based on approximate English frequencies)
    prior_probs = {
        "word": 0.3,   # Common word
        "world": 0.25, # Very common
        "work": 0.2,   # Common
        "worm": 0.15,  # Less common
        "worth": 0.1   # Less common
    }

    # Misspelled word s (transposition error from "word")
    s = "wrod"

    # Manually generated candidates (must be in V)
    candidates = ["word", "world", "work"]

    # Correct the spelling
    correction = noisy_channel_corrector(s, candidates, V, prior_probs)

    print(f"Misspelled word: {s}")
    print(f"Candidates: {candidates}")
    print(f"Dictionary: {V}")
    print(f"Prior probabilities: {prior_probs}")
    print(f"Corrected word: {correction}")


if __name__ == "__main__":
    main()
