"""Rule-based English tokenizer.

Requirements implemented:
1. Punctuation & special symbols become separate tokens.
2. Contractions like "isn't" -> ["is", "n't"] (Treebank style; e.g. "can't" -> ["ca", "n't"].)
3. Abbreviations comprised of ALL CAPS letters (e.g. USA, NATO) kept as single tokens.
4. Internal hyphenated words (e.g. ice-cream, twenty-one) kept as single tokens.

Usage:
	python main.py [optional_path_to_input_file]

If no path is provided it defaults to "input.txt" in the same directory.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Iterable


# Precompile a regex that captures tokens according to the required precedence.
# Order matters: more specific / longer multi-char tokens should appear first.
TOKEN_PATTERN = re.compile(
	r"""
	(?:[A-Z]{2,})                                  # 1. Abbreviations (ALL CAPS, length>=2)
	| (?:[A-Za-z]+(?:-[A-Za-z]+)+)                 # 2. Hyphenated words (letters-hyphen-letters)
	| (?:[A-Za-z]+(?=n't\b))                      # 3. Base of negative contractions (is in isn't, ca in can't)
	| (?:n't\b)                                   # 4. The contracted n't ending
	| (?:[A-Za-z]+(?='(?:re|ve|ll|d|s|m)\b))      # 5. Base before other common endings ('s, 're, ...)
	| (?:'(?:re|ve|ll|d|s|m)\b)                   # 6. Contraction endings themselves
	| (?:\d+(?:-\d+)*)                           # 7. Numbers (optionally hyphenated)
	| (?:[A-Za-z]+)                                # 8. Plain words
	| (?:[^\sA-Za-z0-9])                          # 9. Any single non-space, non-alnum char (punct / symbol)
	""",
	re.VERBOSE,
)


def tokenize(text: str) -> List[str]:
	"""Tokenize input English text per the specified heuristic rules.

	Steps:
	  1. Use a master regex to extract tokens preserving order.
	  2. Filter out stray whitespace (regex avoids matching it anyway).
	"""
	tokens = TOKEN_PATTERN.findall(text)
	return [t for t in tokens if t and not t.isspace()]


def iter_file_text(path: Path) -> Iterable[str]:
	"""Yield file contents as a single string; small helper for clarity."""
	yield path.read_text(encoding="utf-8", errors="replace")


def main(argv: List[str]) -> int:
	if len(argv) > 1:
		input_path = Path(argv[1])
	else:
		input_path = Path(__file__).with_name("input.txt")

	if not input_path.exists():
		print(f"Input file not found: {input_path}", file=sys.stderr)
		return 1

	text = input_path.read_text(encoding="utf-8", errors="replace")
	tokens = tokenize(text)

	# Print one token per line for easy inspection.
	for tok in tokens:
		print(tok)

	return 0


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main(sys.argv))
