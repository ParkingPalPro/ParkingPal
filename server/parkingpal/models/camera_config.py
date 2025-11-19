from server.parkingpal.extensions import db
from datetime import datetime, UTC


class CameraConfig(db.Model):
    __tablename__ = "camera_configs"

    camera_id = db.Column(db.String, primary_key=True)
    config_json = db.Column(db.Text, nullable=False)
    image_width = db.Column(db.Integer, nullable=True)
    image_height = db.Column(db.Integer, nullable=True)
    last_updated = db.Column(db.DateTime, nullable=False, default=datetime.now(UTC))