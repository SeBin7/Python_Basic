import re
import string

from bigram_lm.ngram_getter import NgramGetter

with open("corpus.txt", "r", encoding="utf-8") as file:
    corpus = file.read()


# Step 1: Split the corpus into sentences using punctuation (., !, ?)
sentences = re.split(r"(?<=[.!?])\s+", corpus)


ngram_gettr = NgramGetter()
bigram_model = ngram_gettr.get_ngram(sentences)

def get_prediction(user_input: str) :
    next_word_data = bigram_model[user_input]
        total_occurrences = sum(next_word_data.values())
        # Find the most common next word
        predicted_word = None
        max_count = 0
        for word, count in next_word_data.items():
            if count > max_count:
                max_count = count
                predicted_word = word
        probability = (max_count / total_occurrences) * 100
        
    return predicted_word, probability

# Prediction loop: prompt the user for a word and predict the next word
while True:
    user_input = input("Enter a word (or 'exit' to quit): ").lower().strip()
    if user_input == "exit":
        break
    if user_input in bigram_model:
        predicted_word, probability = get_prediction(user_input)
        print(
            f'Predicted next word: "{predicted_word}" with {probability:.2f}% probability.'
        )
    else:
        print("None")
