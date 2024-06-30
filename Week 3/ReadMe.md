# WEEK 3 TASKS

This repository contains solutions for a series of Python programming tasks focusing on class creation, file handling, package design, and iterators/generators. The tasks are organized into four main areas:

## Task 1: Book and Ebook Classes

This task involves creating a class `Book` with attributes `title`, `author`, and `pages`, along with methods to get and set these attributes. Additionally, a subclass `Ebook` is created that inherits from `Book` and adds an attribute `format`. The `__str__()` method is overridden to display all attributes.

### Features:
- `Book` class with title, author, and pages attributes
- Getter and setter methods for `Book` attributes
- A class method to calculate reading time based on an assumed reading speed
- `Ebook` subclass with an additional format attribute
- Overridden `__str__()` method in `Ebook`

### Files:
- `Library Management System/LMS.py`

## Task 2: File Handling

This task includes reading data from a text file and printing its contents, handling exceptions such as file not found, and errors while reading. Additionally, it includes writing user input to a new file and counting the number of words in the read file.

### Features:
- Read data from a text file and print its contents
- Exception handling for file operations
- Function to write user input to a new file
- Word count functionality for the read file

### Files:
- `file handling.py`

## Task 3: Math Package (mathpkg)

This task involves designing a Python package named `mathpkg` that includes modules for basic arithmetic operations (addition, subtraction, multiplication, division, modulus) and some advanced mathematical operations (exponentiation and square root). A main script demonstrates importing and using functions from this package.

### Features:
- Python package with modules for basic and advanced math operations
- Functions for addition, subtraction, multiplication, division, modulus, exponentiation, and square root
- Main script demonstrating usage of package functions

### Files:
- `mathpkg/` (directory structure for the package)
- `main.py`

## Task 4: Iterators and Generators

This task involves creating an iterator class `Countdown` that counts down from a given number to 1. It also includes generator functions for yielding Fibonacci numbers up to a specified limit and generating a sequence of random numbers within a specified range. A Python program demonstrates the usage of these iterators and generators.

### Features:
- `Countdown` iterator class
- `fibonacci_generator`

