from datetime import datetime, timedelta

from app.database import Base, engine, SessionLocal
from app.models import Order

SAMPLE_ORDERS = [
    {
        "order_id": "ORD-1001",
        "customer_email": "alice@example.com",
        "item_name": "Wireless Headphones",
        "status": "shipped",
        "tracking_number": "TRK-88291",
        "carrier": "FedEx",
        "order_date": datetime.utcnow() - timedelta(days=3),
        "estimated_delivery": datetime.utcnow() + timedelta(days=2),
        "total_amount": 79.99,
    },
    {
        "order_id": "ORD-1002",
        "customer_email": "bob@example.com",
        "item_name": "Yoga Mat",
        "status": "delivered",
        "tracking_number": "TRK-19273",
        "carrier": "UPS",
        "order_date": datetime.utcnow() - timedelta(days=10),
        "estimated_delivery": datetime.utcnow() - timedelta(days=5),
        "total_amount": 24.50,
    },
    {
        "order_id": "ORD-1003",
        "customer_email": "carol@example.com",
        "item_name": "Bluetooth Speaker",
        "status": "processing",
        "tracking_number": None,
        "carrier": None,
        "order_date": datetime.utcnow() - timedelta(days=1),
        "estimated_delivery": datetime.utcnow() + timedelta(days=6),
        "total_amount": 49.99,
    },
    {
        "order_id": "ORD-1004",
        "customer_email": "dave@example.com",
        "item_name": "Desk Lamp",
        "status": "cancelled",
        "tracking_number": None,
        "carrier": None,
        "order_date": datetime.utcnow() - timedelta(days=7),
        "estimated_delivery": None,
        "total_amount": 34.99,
    },
]


def seed():
    """Create tables if missing and seed sample orders once."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.query(Order).count() > 0:
            return
        for data in SAMPLE_ORDERS:
            db.add(Order(**data))
        db.commit()
        print(f"Seeded {len(SAMPLE_ORDERS)} sample orders (try ORD-1001 .. ORD-1004).")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
