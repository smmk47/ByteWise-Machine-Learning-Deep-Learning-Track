import csv
from .inventory import Inventory
from .food_item import FoodItem

class FileManager:
    def __init__(self, filename):
        self.filename = filename

    def read_inventory(self):
        inventory = Inventory()
        try:
            with open(self.filename, mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row:
                        item = FoodItem(*row)
                        inventory.add_item(item)
        except FileNotFoundError:
            print(f"File {self.filename} not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
        return inventory

    def write_inventory(self, inventory):
        try:
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                for item in inventory.items:
                    writer.writerow([item.name, item.category, item.quantity, item.barcode, item.expiry_date])
        except Exception as e:
            print(f"An error occurred: {e}")
