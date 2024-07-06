# Inventory Management Package

## Overview
The `inventory_management` package contains modules that collectively implement an Inventory Management System in Python. This system is designed to manage and organize information about food items in stock, providing various functionalities for CRUD operations, file handling, exception handling, iterators, generators, search, and reporting.

## Modules
The package includes the following modules:

### `__init__.py`
- Initializes the package as a Python module.
- Imports necessary classes and functions.

### `food_item.py`
- Defines the `FoodItem` class.
- Represents individual food items with attributes such as name, category, quantity, barcode, and expiry date.
- Provides methods for initializing, updating, and retrieving item details.

### `inventory.py`
- Defines the `Inventory` class.
- Manages the overall inventory operations.
- Implements methods for adding, editing, deleting, and searching food items.
- Includes functionality for handling near-expiry items, sorting, filtering, and generating reports.

### `file_manager.py`
- Provides file handling operations.
- Implements methods to read from and write to a CSV file.
- Includes error checking and exception handling for file operations.
- Ensures data persistence and integrity through file storage.

## Usage
To utilize the `inventory_management` package in your application:

1. **Import Required Classes**
   ```python
   from inventory_management import FoodItem, Inventory, FileManager

