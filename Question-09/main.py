class PluralNounFSA:
    """
    FSA to accept English plural nouns ending with 'y'.
    Rules:
    - Vowel + y + s (e.g., boys, toys)
    - Consonant + ies (e.g., ponies, skies, puppies)
    - Reject: Consonant + ys (e.g., ponys)
    - Reject: Vowel + ies (e.g., boies, toies)
    """
    
    def __init__(self):
        self.vowels = set('aeiou')
        self.consonants = set('bcdfghjklmnpqrstvwxyz')
    
    def accepts(self, word):
        """Check if word is accepted by the FSA."""
        word = word.lower()
        
        if len(word) < 3:
            return False
        
        if word.endswith('ys'):
            if len(word) >= 3 and word[-3] in self.vowels:
                return True
            else:
                return False
        
        elif word.endswith('ies'):
            if len(word) >= 4 and word[-4] in self.consonants:
                return True
            else:
                return False
        
        return False
    
    def trace(self, word):
        """Show FSA state transitions for the word."""
        word = word.lower()
        print(f"\nTracing FSA for: '{word}'")
        print("-" * 50)
        
        if len(word) < 3:
            print("State: q0 (start)")
            print("Result: REJECT (word too short)")
            return False
        
        print("State: q0 (start)")
        print(f"Reading prefix: '{word[:-3] if len(word) > 3 else ''}'")
        
        if word.endswith('ys'):
            print(f"State: q1 (reading letters before 'ys')")
            char_before = word[-3]
            print(f"Character before 'ys': '{char_before}'")
            
            if char_before in self.vowels:
                print(f"State: q2 (vowel + 'y')")
                print(f"State: q3 (vowel + 'ys')")
                print("State: q_accept (ACCEPT)")
                return True
            else:
                print(f"State: q4 (consonant + 'ys')")
                print("State: q_reject (REJECT - consonant + ys not allowed)")
                return False
        
        elif word.endswith('ies'):
            print(f"State: q1 (reading letters before 'ies')")
            if len(word) >= 4:
                char_before = word[-4]
                print(f"Character before 'ies': '{char_before}'")
                
                if char_before in self.consonants:
                    print(f"State: q5 (consonant + 'i')")
                    print(f"State: q6 (consonant + 'ie')")
                    print(f"State: q7 (consonant + 'ies')")
                    print("State: q_accept (ACCEPT)")
                    return True
                else:
                    print(f"State: q8 (vowel + 'ies')")
                    print("State: q_reject (REJECT - vowel + ies not allowed)")
                    return False
            else:
                print("State: q_reject (REJECT - word too short)")
                return False
        
        else:
            print("State: q_reject (REJECT - doesn't end with 'ys' or 'ies')")
            return False


def test_fsa():
    """Test the FSA with various examples."""
    fsa = PluralNounFSA()
    
    test_cases = [
        # Should accept
        ("boys", True),
        ("toys", True),
        ("ponies", True),
        ("skies", True),
        ("puppies", True),
        ("days", True),
        ("keys", True),
        ("ladies", True),
        ("babies", True),
        ("cities", True),
        
        # Should reject
        ("boies", False),
        ("toies", False),
        ("ponys", False),
        ("skys", False),
        ("puppys", False),
        ("boy", False),
        ("pony", False),
        ("cat", False),
        ("cats", False)
    ]
    
    print("FINITE STATE AUTOMATA: English Plural Nouns Ending with 'y'")
    print("\nRules:")
    print("  1. Vowel + y → Vowel + ys (e.g., boy → boys)")
    print("  2. Consonant + y → Consonant + ies (e.g., pony → ponies)")
    
    print("\nTEST RESULTS:")
    
    correct = 0
    for word, expected in test_cases:
        result = fsa.accepts(word)
        status = "✓" if result == expected else "✗"
        correct += (result == expected)
        
        print(f"{status} '{word:15}' → {'ACCEPT' if result else 'REJECT':8} (Expected: {'ACCEPT' if expected else 'REJECT'})")
    
    print(f"Accuracy: {correct}/{len(test_cases)} ({100*correct//len(test_cases)}%)")
    
    print("\nDETAILED FSA TRACES:")
    
    for word in ["boys", "ponies", "ponys", "boies"]:
        fsa.trace(word)


if __name__ == "__main__":
    test_fsa()