class PredictionGetter:
    def get_prediction(user_input: str, bigram_model: str) :
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
        