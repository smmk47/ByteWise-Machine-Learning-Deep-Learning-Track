
class Countdown:
    
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.start < 1:
            raise StopIteration
        current = self.start
        self.start -= 1
        return current

def fibonacci_generator(limit):
    
    a, b = 0, 1
    while a <= limit:
        yield a
        a, b = b, a + b

def random_number_generator(start, end, count):
    
    import random
    for _ in range(count):
        yield random.randint(start, end)

def main():
    try:
        # Using the Countdown iterator
        countdown = Countdown(5)
        for num in countdown:
            print(num)

        # Using the fibonacci_generator
        fib_limit = 100
        fib_gen = fibonacci_generator(fib_limit)
        for num in fib_gen:
            print(num)

        # Using the random_number_generator
        start, end, count = 1, 100, 5
        rand_gen = random_number_generator(start, end, count)
        for num in rand_gen:
            print(num)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()