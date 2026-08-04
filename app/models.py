from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime

from app.database import Base


class Order(Base):
    """Mock order-management table, standing in for a real order system."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String, unique=True, index=True, nullable=False)
    customer_email = Column(String, index=True, nullable=False)
    item_name = Column(String, nullable=False)
    status = Column(String, nullable=False)  # processing | shipped | delivered | cancelled
    tracking_number = Column(String, nullable=True)
    carrier = Column(String, nullable=True)
    order_date = Column(DateTime, default=datetime.utcnow)
    estimated_delivery = Column(DateTime, nullable=True)
    total_amount = Column(Float, nullable=True)


class SupportTicket(Base):
    """Created automatically whenever the agent escalates to a human."""

    __tablename__ = "support_tickets"

    id = Column(Integer, primary_key=True, index=True)
    customer_message = Column(String, nullable=False)
    reason = Column(String, nullable=False)
    intent = Column(String, nullable=True)
    order_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="open")  # open | resolved
