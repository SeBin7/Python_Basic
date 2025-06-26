class PyramidDrawer:
    def draw(self, alphabet: str, n: int) -> str:
        answer = ""
        width = 2 * n - 1
        for i in range(1, n + 1):
            # For each row, calculate the number of characters to print (an odd number)
            num_chars = 2 * i - 1
            # Center the row by adding spaces to the left
            spaces = (width - num_chars) // 2

            answer += " " * spaces + alphabet * num_chars + "\n"
        return answer
