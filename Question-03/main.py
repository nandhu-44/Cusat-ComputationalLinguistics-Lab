import re

def tokenize_text(text):
    """
    Simple rule-based tokenizer for English text using regex.
    Handles punctuation, contractions, abbreviations, and hyphenated words.
    """
    
    # Define tokenization patterns (order matters)
    patterns = [
        # Contractions: n't, 'll, 're, 've, 'd, 's
        r"n't\b",           # isn't -> is + n't
        r"'ll\b",           # I'll -> I + 'll  
        r"'re\b",           # you're -> you + 're
        r"'ve\b",           # I've -> I + 've
        r"'d\b",            # I'd -> I + 'd
        r"'s\b",            # John's -> John + 's
        
        # Abbreviations (2-5 uppercase letters)
        r"\b[A-Z]{2,5}\b",  # USA, NATO, FBI
        
        # Hyphenated words (letters-letters)
        r"\b\w+(?:-\w+)+\b", # ice-cream, twenty-one
        
        # Words (letters, numbers, underscores)
        r"\b\w+\b",
        
        # Punctuation and special symbols (each as separate token)
        r"[^\w\s]"
    ]
    
    # Combine all patterns
    token_pattern = '|'.join(f'({pattern})' for pattern in patterns)
    
    # Find all matches
    matches = re.findall(token_pattern, text)
    
    # Extract non-empty tokens
    tokens = []
    for match in matches:
        for group in match:
            if group:
                tokens.append(group)
                break
    
    return tokens

def tokenize_file(filename):
    """Read text from file and tokenize it."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        return tokenize_text(text)
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return []

# Test the tokenizer
if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    Hello world! I can't believe it's working. 
    The USA and NATO won't cooperate. 
    Ice-cream costs twenty-one dollars.
    Dr. Smith's email: john@example.com (555) 123-4567.
    """
    
    print("Sample Text:")
    print(sample_text)
    print("\nTokens:")
    tokens = tokenize_text(sample_text)
    for i, token in enumerate(tokens, 1):
        print(f"{i:2d}: '{token}'")
    
    # Test with file input
    print("\n" + "="*50)
    print("Testing with input file...")
    
    # Create sample input file
    with open('input.txt', 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    file_tokens = tokenize_file('input.txt')
    print(f"Tokens from file: {len(file_tokens)}")
    for i, token in enumerate(file_tokens, 1):  # Show all tokens
        print(f"{i:2d}: '{token}'")