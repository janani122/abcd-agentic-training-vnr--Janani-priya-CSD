import re

def tokenize(text):
    """
    Splits the input text into tokens (words) using regex.
    Punctuation is removed, and all words are converted to lowercase.
    """
    # Remove punctuation and split by whitespace
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

# Example usage
text = "Hello, world! Welcome to string tokenization."
tokens = tokenize(text)
print(tokens)
