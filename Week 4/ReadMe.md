# Inventory Management System

## Overview
The Inventory Management System is a Python application designed to manage and organize information about food items in stock. It provides various features including CRUD operations, file handling, exception handling, iterators, generators, search, and reporting functionalities.

## Project Structure
The project is structured as follows:
week4/
├── main.py
└── inventory_management/
├── init.py
├── food_item.py
├── inventory.py
└── file_manager.py
- **`main.py`**: Entry point of the application where user interaction takes place.
- **`inventory_management/`**: Package containing all the modules related to inventory management.
  - **`__init__.py`**: Initializes the package.
  - **`food_item.py`**: Defines the `FoodItem` class to represent individual food items.
  - **`inventory.py`**: Defines the `Inventory` class to manage the overall inventory and operations.
  - **`file_manager.py`**: Provides file handling operations to read from and write to CSV files.

## Features
1. **Real-Time Data Handling**
   - CRUD operations for managing inventory data stored in a CSV file.
   - Date conversion and expiry date warnings for items nearing expiration.
2. **Efficient Data Storage**
   - Each `FoodItem` object represents an item with attributes like name, category, quantity, barcode, and expiry date.
   - Date handling ensures accurate comparison and management of expiry dates.
3. **User Interaction**
   - Menu-driven interface in `main.py` allows users to interact with the inventory system.
   - Input validation ensures data integrity, especially for critical operations like barcode entry.
4. **Sorting & Filtering**
   - Items can be sorted based on expiry dates.
   - Filtering options available for categories or other criteria.
5. **Search & Reporting**
   - Search functionality allows users to find items by barcode, name, or category.
   - Reporting features generate detailed reports based on inventory data.

## Usage
To utilize the `inventory_management` package in your application:

1. **Import Required Classes**
   ```python
   from inventory_management import FoodItem, Inventory, FileManager




