from server.parkingpal.extensions import db


class ParkingSession(db.Model):
    __tablename__ = "cars"

    car_id_hash = db.Column(db.String, primary_key=True)
    timestamp_start = db.Column(db.DateTime(timezone=True))
    timestamp_end = db.Column(db.DateTime(timezone=True))

