import re

def extract_digits_and_phones(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    digits = re.findall(r'\d', text)

    phones = re.findall(r'(?:\+91[-\s]?|0)?(\d{10})\b', text)
    phones = list(set(phones))

    result = f"""File: {filename}\n
Extracted Digits         : {digits if digits else 'None'}\n
Extracted Phone Numbers  : {phones if phones else 'None'}\n
Total Digits             : {len(digits)}
Total Phone Numbers      : {len(phones)}"""

    print(result)    
    with open("output.txt", "w", encoding='utf-8') as f:
        f.write(result)

extract_digits_and_phones("input.txt")