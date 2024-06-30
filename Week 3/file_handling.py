def read_file(filename):
    """
    Reads data from a text file and prints its contents.
    """
    try:
        with open(filename, 'r') as file:
            data = file.read()
            print(data)
            return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error: {e}")

def count_words(filename):
    """
    Counts the number of words in a file and prints the result.
    """
    data = read_file(filename)
    if data:
        words = data.split()
        print(f"Number of words: {len(words)}")

def write_file(filename):
    """
    Writes user input to a new file and handles exceptions related to file writing.
    """
    try:
        with open(filename, 'w') as file:
            user_input = input("Enter text to write to file: ")
            file.write(user_input)
            print(f"Data written to {filename}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
read_file('data.txt')
count_words('data.txt')
write_file('output.txt')