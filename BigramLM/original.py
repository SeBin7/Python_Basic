import re
import string


with open("corpus.txt", "r", encoding="utf-8") as file:
    corpus = file.read()


# Step 1: Split the corpus into sentences using punctuation (., !, ?)
sentences = re.split(r"(?<=[.!?])\s+", corpus)

# Step 2: Build the bigram model using plain dictionaries
bigram_model = {}

for sentence in sentences:
    # Preprocess the sentence: lowercase and remove punctuation
    sentence_clean = sentence.lower().translate(
        str.maketrans("", "", string.punctuation)
    )
    tokens = sentence_clean.split()
    # Build bigram counts for the sentence
    for i in range(len(tokens) - 1):
        first_word = tokens[i]
        second_word = tokens[i + 1]
        if first_word not in bigram_model:
            bigram_model[first_word] = {}  # initialize inner dictionary
        if second_word not in bigram_model[first_word]:
            bigram_model[first_word][second_word] = 0
        bigram_model[first_word][second_word] += 1

# Prediction loop: prompt the user for a word and predict the next word
while True:
    user_input = input("Enter a word (or 'exit' to quit): ").lower().strip()
    if user_input == "exit":
        break
    if user_input in bigram_model:
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
        print(
            f'Predicted next word: "{predicted_word}" with {probability:.2f}% probability.'
        )
    else:
        print("None")
