from . import api_bp
from flask import jsonify, request
from datetime import datetime, UTC
from server.parkingpal.extensions import db
from server.parkingpal.models import ParkingSpace



@api_bp.route("/parking/update", methods=["POST"])
def update_parking_spaces():
    try:
        data = request.get_json()
        if not data or "spaces" not in data:
            return jsonify({"error": "Invalid data format"}), 400

        now = datetime.now(UTC)

        for s in data["spaces"]:
            space_number = s.get("space_number")
            is_free = s.get("is_free")

            if space_number is None or is_free is None:
                continue

            # Prevent autoflush from inserting before checking
            with db.session.no_autoflush:
                space = ParkingSpace.query.get(space_number)

            if space is None:
                space = ParkingSpace(
                    space_number=space_number,
                    is_free=is_free,
                    last_updated=now
                )
                db.session.add(space)
            else:
                space.is_free = is_free
                space.last_updated = now

        db.session.commit()
        return jsonify({"success": True, "message": "Parking spaces updated"}), 200

    except Exception as e:
        db.session.rollback()
        print("Error updating parking:", e)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/parking/status", methods=["GET"])
def get_parking_status():
    try:
        spaces = ParkingSpace.query.order_by(ParkingSpace.space_number).all()

        spaces_list = [
            {
                "id": s.space_number,
                "status": "available" if s.is_free else "occupied",
                "last_updated": s.last_updated
            }
            for s in spaces
        ]

        total = len(spaces_list)
        available = sum(1 for s in spaces_list if s["status"] == "available")
        occupied = total - available

        return jsonify({
            "total": total,
            "available": available,
            "occupied": occupied,
            "spaces": spaces_list
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


