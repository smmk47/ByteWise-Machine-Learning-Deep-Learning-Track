from datetime import datetime

class FoodItem:
    def __init__(self, name, category, quantity, barcode, expiry_date):
        self.name = name
        self.category = category
        self.quantity = quantity
        self.barcode = barcode
        self.expiry_date = expiry_date

    def __str__(self):
        return f"{self.name}, {self.category}, {self.quantity}, {self.barcode}, {self.expiry_date}"

    def is_near_expiry(self, days=7):
        current_date = datetime.now().date()
        expiry_date = datetime.strptime(self.expiry_date, "%Y-%m-%d").date()
        return (expiry_date - current_date).days <= days
