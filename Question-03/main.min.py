import re

def tokenize(text):
    tokens = []
    words = text.split()
    for word in words:
        if word.lower().endswith("n't"):
            base = word[:-3]
            if word.lower() == "can't":
                tokens.extend(["ca", "n't"])
            elif word.lower() == "won't":
                tokens.extend(["wo", "n't"])
            else:
                tokens.extend([base, "n't"])
            continue
        
        if "'" in word:
            parts = word.split("'")
            tokens.append(parts[0])
            tokens.append("'" + parts[1])
            continue
        
        if re.match(r'^[A-Z]+$', word.rstrip('.,!?;:')):
            match = re.match(r'^([A-Z]+)([.,!?;:]*)$', word)
            if match:
                tokens.append(match.group(1))
                tokens.extend(list(match.group(2)))
            continue
        
        if re.match(r'^[\w]+-[\w]+', word):
            match = re.match(r'^([\w]+-[\w]+)([.,!?;:]*)$', word)
            if match:
                tokens.append(match.group(1))
                tokens.extend(list(match.group(2)))
            continue
        
        current = ""
        for char in word:
            if char in '.,!?;:':
                if current:
                    tokens.append(current)
                    current = ""
                tokens.append(char)
            else:
                current += char
        
        if current:
            tokens.append(current)
    
    return tokens

if __name__ == "__main__":
    input_path = "input.txt"
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
        lines = text.splitlines()
        for line in lines:
            print(f"Input:  {line}")
            print(f"Tokens: {tokenize(line)}")
            print()