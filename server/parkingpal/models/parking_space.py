from server.parkingpal.extensions import db
from datetime import datetime, UTC


class ParkingSpace(db.Model):
    __tablename__ = "parking_spaces"

    id = db.Column(db.Integer, primary_key=True)
    space_number = db.Column(db.Integer, unique=True, nullable=False)
    is_free = db.Column(db.Boolean, default=True, nullable=False)
    last_updated = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(UTC))
