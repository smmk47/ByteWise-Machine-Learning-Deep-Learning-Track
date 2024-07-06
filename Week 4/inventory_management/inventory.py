from datetime import datetime
from .food_item import FoodItem

class Inventory:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)

    def edit_item(self, barcode, **kwargs):
        item = self.search_item_by_barcode(barcode)
        if item:
            for key, value in kwargs.items():
                setattr(item, key, value)
        else:
            print("Item not found!")

    def delete_item(self, barcode):
        item = self.search_item_by_barcode(barcode)
        if item:
            self.items.remove(item)
        else:
            print("Item not found!")

    def search_item_by_barcode(self, barcode):
        for item in self.items:
            if item.barcode == barcode:
                return item
        return None

    def search_item_by_name(self, name):
        return [item for item in self.items if item.name.lower() == name.lower()]

    def get_near_expiry_items(self, days=7):
        return [item for item in self.items if item.is_near_expiry(days)]

    def __str__(self):
        return "\n".join(str(item) for item in self.items)

    def __iter__(self):
        return iter(self.items)

    def near_expiry_generator(self, days=7):
        for item in self.items:
            if item.is_near_expiry(days):
                yield item

    def generate_report(self):
        report = {
            "total_items": len(self.items),
            "near_expiry_items": len(self.get_near_expiry_items()),
            "category_summary": self.get_category_summary()
        }
        return report

    def get_category_summary(self):
        summary = {}
        for item in self.items:
            if item.category in summary:
                summary[item.category] += 1
            else:
                summary[item.category] = 1
        return summary
