# main.py

from mathpkg.addition import add
from mathpkg.subtraction import subtract
from mathpkg.multiplication import multiply
from mathpkg.division import divide
from mathpkg.modulus import modulus
from mathpkg.exponentiation import power
from mathpkg.square_root import square_root

def main():
    try:
        # Demonstrate addition
        print("Addition: 5 + 3 =", add(5, 3))
        
        # Demonstrate subtraction
        print("Subtraction: 5 - 3 =", subtract(5, 3))
        
        # Demonstrate multiplication
        print("Multiplication: 5 * 3 =", multiply(5, 3))
        
        # Demonstrate division
        print("Division: 5 / 3 =", divide(5, 3))
        
        # Demonstrate modulus
        print("Modulus: 5 % 3 =", modulus(5, 3))
        
        # Demonstrate exponentiation
        print("Exponentiation: 5 ** 3 =", power(5, 3))
        
        # Demonstrate square root
        print("Square Root: sqrt(25) =", square_root(25))

        # Handle potential errors
    except ZeroDivisionError as e:
        print(e)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
