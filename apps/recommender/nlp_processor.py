import re
import string
import os
import pandas as pd
from typing import List, Dict

# Global lexicon cache
LEXICON: Dict[str, int] = {}
LEXICON_LOADED = False

# Common Turkish stopwords
TURKISH_STOPWORDS = {
    "bir", "mi", "mı", "mu", "mü", "o", "da", "de", "ki", "ve", "ile",
    "gibi", "için", "çok", "bu", "şu", "her", "ben", "sen", "biz", "siz",
    "onlar", "var", "yok", "en", "çok", "daha", "ancak", "sanki", "göre",
    "ise", "neden", "ancak", "bile", "hiç", "ama", "fakat", "eğer", "gibi",
    "ise", "sadece", "da", "ki", "zaten"
}

def load_lexicon():
    """
    Loads the SWNetTR++ lexicon from the CSV file.
    """
    global LEXICON, LEXICON_LOADED
    if LEXICON_LOADED:
        return

    try:
        # Resolve path relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "data", "SWNetTR++.csv")
        
        # Detect header line dynamically
        header_line_index = None
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            for i, line in enumerate(f):
                if "WORD;TONE;POLARITY" in line:
                    header_line_index = i
                    break
        
        if header_line_index is None:
            print("❌ Error: Could not find header 'WORD;TONE;POLARITY' in SWNetTR++.csv")
            return

        # Read CSV with semi-colon separator, starting from the detected header
        # Added decimal=',' because the file uses commas for decimals (e.g. -0,125)
        df = pd.read_csv(csv_path, sep=';', header=0, skiprows=header_line_index, encoding="utf-8-sig", on_bad_lines='skip', decimal=',')
        
        # Normalize columns just in case
        df.columns = [c.strip().upper() for c in df.columns]
        
        if 'WORD' not in df.columns or 'POLARITY' not in df.columns:
            print(f"❌ Error: Expected columns WORD and POLARITY not found. Found: {df.columns}")
            return

        for _, row in df.iterrows():
            word = str(row['WORD']).strip().lower()
            try:
                polarity = int(row['POLARITY'])
                LEXICON[word] = polarity
            except ValueError:
                continue
                
        LEXICON_LOADED = True
        print(f"✅ Loaded {len(LEXICON)} words from sentiment lexicon.")

    except Exception as e:
        print(f"❌ Error loading sentiment lexicon: {e}")

def preprocess_text(text: str) -> List[str]:
    """
    Cleans and tokenizes Turkish text.
    """
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = text.split()
    # Remove stopwords
    tokens = [word for word in tokens if word not in TURKISH_STOPWORDS]
    
    return tokens


def analyze_sentiment(reviews_text: str) -> float:
    """
    Performs sentiment analysis using SWNetTR++ lexicon.
    """
    if not LEXICON_LOADED:
        load_lexicon()

    if not reviews_text or reviews_text.strip() == "":
        return 0.0

    # Split into individual reviews (if concatenated by " ||| ")
    individual_reviews = reviews_text.split(" ||| ")
    
    total_sentiment = 0.0
    review_count = 0

    for review in individual_reviews:
        if not review.strip():
            continue

        tokens = preprocess_text(review)
        
        review_score = 0
        for token in tokens:
            # 1. Exact match
            if token in LEXICON:
                review_score += LEXICON[token]
                continue
            
            # 2. Substring/Root match (Simple approach)
            # Check if any lexicon word is a prefix of the token (common in Turkish)
            # We filter for reasonable length to avoid matching "a" or "i"
            # This is O(N*M) but acceptable for small-medium text
            # Optimization: Check only keys that start with the same letter could be done, 
            # but for now we iterate common roots if needed. 
            # Better approach: Reverse check. Is the token a superstring of a key?
            
            # NOTE: Iterating whole lexicon per token is too slow.
            # Instead, we rely on exact match + basic stemming heuristics or 
            # partial matches if we had a trie. 
            # For this iteration, we stick to EXACT match + simple suffix stripping.
            
            # Simple stemming helper
            suffixes = ["ler", "lar", "nin", "nın", "den", "dan", "yi", "yı", "yu", "yü", "de", "da", "di", "dı", "miş", "mış", "yor", "du"]
            found = False
            for suffix in suffixes:
                if token.endswith(suffix):
                    stem = token[:-len(suffix)]
                    if stem in LEXICON:
                        review_score += LEXICON[stem]
                        found = True
                        break
            if found:
                continue

        # Normalize review score to -1..1 range loosely
        if review_score > 0:
            total_sentiment += 1.0
        elif review_score < 0:
            total_sentiment -= 1.0
        
        review_count += 1

    if review_count == 0:
        return 0.0

    return total_sentiment / review_count

def get_sentiment_label(score: float) -> str:
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

