class AlphabetCounter:
    def count(self, s: str) -> tuple[str, int]:
        max_char = ""
        max_count = 0
        cur_char = ""
        cur_count = 0

        for char in s:
            if char == cur_char:
                cur_count += 1
            else:
                cur_char = char
                cur_count = 1
            if cur_count > max_count:
                max_count = cur_count
                max_char = char
        return max_char, max_count
