from inventory_management import FoodItem, Inventory, FileManager

def main():
    file_manager = FileManager("inventory.csv")
    inventory = file_manager.read_inventory()

    while True:
        print("\nInventory Management System")
        print("1. Add Item")
        print("2. Edit Item")
        print("3. Delete Item")
        print("4. Search Item by Barcode")
        print("5. Search Item by Name")
        print("6. Display Near Expiry Items")
        print("7. Generate Report")
        print("8. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            name = input("Enter name: ")
            category = input("Enter category: ")
            quantity = input("Enter quantity: ")
            barcode = input("Enter barcode: ")
            expiry_date = input("Enter expiry date (YYYY-MM-DD): ")
            item = FoodItem(name, category, quantity, barcode, expiry_date)
            inventory.add_item(item)
            file_manager.write_inventory(inventory)

        elif choice == "2":
            barcode = input("Enter barcode of the item to edit: ")
            name = input("Enter new name (leave blank to keep current): ")
            category = input("Enter new category (leave blank to keep current): ")
            quantity = input("Enter new quantity (leave blank to keep current): ")
            expiry_date = input("Enter new expiry date (YYYY-MM-DD) (leave blank to keep current): ")

            kwargs = {}
            if name:
                kwargs['name'] = name
            if category:
                kwargs['category'] = category
            if quantity:
                kwargs['quantity'] = quantity
            if expiry_date:
                kwargs['expiry_date'] = expiry_date

            inventory.edit_item(barcode, **kwargs)
            file_manager.write_inventory(inventory)

        elif choice == "3":
            barcode = input("Enter barcode of the item to delete: ")
            inventory.delete_item(barcode)
            file_manager.write_inventory(inventory)

        elif choice == "4":
            barcode = input("Enter barcode to search: ")
            item = inventory.search_item_by_barcode(barcode)
            if item:
                print(item)
            else:
                print("Item not found!")

        elif choice == "5":
            name = input("Enter name to search: ")
            items = inventory.search_item_by_name(name)
            if items:
                for item in items:
                    print(item)
            else:
                print("Item not found!")

        elif choice == "6":
            days = int(input("Enter number of days to check for near expiry items: "))
            items = inventory.get_near_expiry_items(days)
            if items:
                for item in items:
                    print(item)
            else:
                print("No items near expiry.")

        elif choice == "7":
            report = inventory.generate_report()
            print("Inventory Report:")
            print(f"Total Items: {report['total_items']}")
            print(f"Near Expiry Items: {report['near_expiry_items']}")
            print("Category Summary:")
            for category, count in report['category_summary'].items():
                print(f"{category}: {count}")

        elif choice == "8":
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
