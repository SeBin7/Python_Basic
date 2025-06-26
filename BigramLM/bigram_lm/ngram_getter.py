import string

class NgramGetter:
    def get_ngram(self, sentences: list[str]) -> dict:
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
                
        return bigram_model