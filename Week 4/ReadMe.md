# Inventory Management System

## Overview
The Inventory Management System is a Python application designed to help manage and organize information about food items in stock. The system supports various features including CRUD operations, file handling, exception handling, iterators, generators, search, and reporting functionalities.

## Key Features
1. **Real-Time Data Handling**: 
   - Manage CRUD (Create, Read, Update, Delete) operations for inventory data stored in a CSV file.
   - Handle date conversion and display warnings for expired or soon-to-expire items.
2. **Efficient Data Storage**: 
   - Represent individual food items with attributes such as name, category, quantity, barcode, and expiry date.
   - Store and compare date and time information effectively.
3. **User Interaction**: 
   - Provide a user-friendly, menu-driven interface.
   - Ensure valid input for critical operations like barcode entry.
4. **Sorting & Filtering**: 
   - Prioritize items based on expiry dates.
   - Filter items by category or other criteria.
5. **Search & Reporting**: 
   - Implement search functionality to find items by barcode or name.
   - Generate reports based on inventory data.

## Development Stages
### Stage 1: Object-Oriented Programming (OOP)
- **Tasks**:
  - Create the `FoodItem` class to represent individual food items.
  - Create the `Inventory` class to manage the overall inventory, including core operations.
  - Implement methods for adding, editing, deleting, and searching food items.
  - Implement methods for handling near-expiry items.
- **Deliverables**:
  - Python files defining the `FoodItem` and `Inventory` classes with appropriate attributes and methods.
  - Methods for searching and handling near-expiry items.

### Stage 2: Files & Exception Handling
- **Tasks**:
  - Implement methods to read and write inventory data to a CSV file.
  - Add error checking and handling for file operations and user inputs.
- **Deliverables**:
  - Updated `Inventory` class with methods for reading and writing to a CSV file.
  - Exception handling integrated into file operations and user inputs.

### Stage 3: Modularize the Project
- **Tasks**:
  - Split your code into multiple modules (e.g., `food_item.py`, `inventory.py`, `file_manager.py`).
  - Create a package and include an `__init__.py` file.
- **Deliverables**:
  - A Python package containing the modules for the Inventory Management System.

### Stage 4: Iterators & Generators
- **Tasks**:
  - Create an iterator to iterate through food items in the inventory.
  - Create a generator to yield food items that are nearing their expiry date.
- **Deliverables**:
  - Updated the `Inventory` class with iterator and generator methods.
  - Examples demonstrating the use of iterators and generators.

### Stage 5: Advanced Features (Search and Reporting)
- **Tasks**:
  - Implement methods to search for food items by barcode, name, or category.
  - Create methods to generate reports based on inventory data, such as items nearing expiry, items in low stock, and category-based summaries.
- **Deliverables**:
  - Fully functional `Inventory` class with search and reporting features.
  - Sample scripts demonstrating the use of advanced features.

## Project Structure
