from .alphabet_counter import AlphabetCounter
from .pyramid_drawer import PyramidDrawer


class AlphabetPyramid:
    def __init__(self) -> None:
        self.counter = AlphabetCounter()
        self.drawer = PyramidDrawer()

    def run(self) -> None:
        while True:
            s = input("Enter a string: ")
            max_char, max_count = self.counter.count(s)
            answer = self.drawer.draw(max_char, max_count)
            print(answer)
