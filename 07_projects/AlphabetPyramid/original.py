# Get input string from the user
s = input("Enter a string: ")

# Initialize variables to track the longest consecutive sequence
max_char = ""
max_count = 0
cur_char = ""
cur_count = 0

# Loop through each character in the string
for char in s:
    if char == cur_char:
        cur_count += 1
    else:
        cur_char = char
        cur_count = 1
    if cur_count > max_count:
        max_count = cur_count
        max_char = char

# Print the longest consecutive character and its count
print(f"{max_char}, {max_count}")


answer = ""
# Draw a pyramid using the longest consecutive character.
# The pyramid has 'max_count' lines and its base width is (2 * max_count - 1)
width = 2 * max_count - 1
for i in range(1, max_count + 1):
    # For each row, calculate the number of characters to print (an odd number)
    num_chars = 2 * i - 1
    # Center the row by adding spaces to the left
    spaces = (width - num_chars) // 2
    answer += " " * spaces + max_char * num_chars + "\n"

print(answer)
