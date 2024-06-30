class Book:
    def __init__(self, title, author, pages):
        """
        Initializes a Book object with title, author, and pages.
        """
        self._title = title
        self._author = author
        self._pages = pages

    def get_title(self):
        """Getter method for title"""
        return self._title

    def set_title(self, title):
        """Setter method for title"""
        self._title = title

    def get_author(self):
       
        return self._author

    def set_author(self, author):
       
        self._author = author

    def get_pages(self):
        
        return self._pages

    def set_pages(self, pages):
        
        self._pages = pages

    @classmethod
    def calculate_reading_time(cls, pages, reading_speed=250):
        """
        Calculates the reading time in minutes based on the number of pages and reading speed.
        """
        return pages / reading_speed

    def __str__(self):
        """
        Returns a string representation of the Book object.
        """
        return f"Title: {self._title}, Author: {self._author}, Pages: {self._pages}"


class Ebook(Book):
    def __init__(self, title, author, pages, format):
        """
        Initializes an Ebook object with title, author, pages, and format.
        """
        super().__init__(title, author, pages)
        self._format = format

    def get_format(self):
        """Getter method for format"""
        return self._format

    def set_format(self, format):
        """Setter method for format"""
        self._format = format

    def __str__(self):
        """
        Returns a string representation of the Ebook object.
        """
        return f"{super().__str__()}, Format: {self._format}"


book = Book("To Kill a Mockingbird", "Harper Lee", 281)
print(book)
print(f"Reading time: {Book.calculate_reading_time(book.get_pages())} minutes")

ebook = Ebook("The Great Gatsby", "F. Scott Fitzgerald", 180, "EPUB")
print(ebook)
print(f"Reading time: {Book.calculate_reading_time(ebook.get_pages())} minutes")