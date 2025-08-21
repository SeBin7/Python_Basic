from alphabet_pyramid import AlphabetCounter


def test_alphabet_pyramid():
    s = "aaabbbcc"
    counter = AlphabetCounter()
    max_char, max_count = counter.count(s)
    assert max_char == "a"
    assert max_count == 3
