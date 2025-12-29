import re
import string
from typing import List

# Simple Turkish sentiment lexicon (can be expanded)
POSITIVE_WORDS = [
    "harika", "mükemmel", "çok iyi", "enfes", "lezzetli", "güzel",
    "şahane", "nefis", "bayıldım", "tertemiz", "taze", "başarılı",
    "sevdim", "kaliteli", "hızlı", "ilgili", "güler yüzlü", "memnun",
    "beğendim", "şiddetle tavsiye", "harikulade", "profesyonel",
    "sakin", "huzurlu", "güvenilir", "eşsiz", "doyurucu", "ucuz", "uygun"
]

NEGATIVE_WORDS = [
    "kötü", "berbat", "rezalet", "vasat", "sıkıcı", "yavaş", "ilgisiz",
    "pahalı", "hayal kırıklığı", "vasatın altında", "tatsız", "bayat",
    "kirli", "soğuk", "saygısız", "kaba", "çirkin", "sorun", "şikayet",
    "fiyasko", "lezzetsiz", "pişman", "korkunç", "düşük", "uygunsuz",
    "vasatın altında", "çok pahalı", "düş kırıklığı"
]

# Common Turkish stopwords (can be expanded)
# This list is a basic example. For a real application, consider using NLTK's
# Turkish stopwords or a more comprehensive list.
TURKISH_STOPWORDS = {
    "bir", "mi", "mı", "mu", "mü", "o", "da", "de", "ki", "ve", "ile",
    "gibi", "için", "çok", "bu", "şu", "her", "ben", "sen", "biz", "siz",
    "onlar", "var", "yok", "en", "çok", "daha", "ancak", "sanki", "göre",
    "ise", "neden", "ancak", "bile", "hiç", "ama", "fakat", "eğer", "gibi",
    "ise", "sadece", "da", "ki", "zaten"
}


def preprocess_text(text: str) -> List[str]:
    """
    Cleans and tokenizes Turkish text.
    - Converts to lowercase.
    - Removes punctuation and numbers.
    - Removes extra whitespace.
    - Removes stopwords.
    """
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = text.split()
    # Remove stopwords
    tokens = [word for word in tokens if word not in TURKISH_STOPWORDS]
    
    return tokens


def analyze_sentiment(reviews_text: str) -> float:
    """
    Performs a simple lexicon-based sentiment analysis on concatenated review texts.
    Returns a score between -1 (very negative) and 1 (very positive).
    """
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
        
        positive_count = sum(1 for word in tokens if word in POSITIVE_WORDS)
        negative_count = sum(1 for word in tokens if word in NEGATIVE_WORDS)
        
        # Simple sentiment calculation for this review
        if positive_count > negative_count:
            total_sentiment += 1.0
        elif negative_count > positive_count:
            total_sentiment -= 1.0
        # Else, neutral (adds 0.0)
        
        review_count += 1

    if review_count == 0:
        return 0.0

    # Average sentiment across all reviews for this restaurant
    return total_sentiment / review_count

def get_sentiment_label(score: float) -> str:
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

