import re

from .prediction_getter import Predicter
from .ngram_getter import NgramGetter


class BigramLM:
    def __init__(self) -> None:
        self.predicter = Predicter()
        self.ngram = NgramGetter()
        
    def run(self):
        with open("corpus.txt", "r", encoding="utf-8") as file:
            corpus = file.read()

        sentences = re.split(r"(?<=[.!?])\s+", corpus)
        ngram = self.ngram_getter.get_ngram(sentences)

