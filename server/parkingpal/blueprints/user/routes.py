from . import user_bp
from server.parkingpal.utils import login_required, hash_plate_number, calculate_duration
from server.parkingpal.models.parking_session import ParkingSession
from server.parkingpal.extensions import db
from flask import redirect, url_for, session, render_template, request, jsonify
from datetime import datetime, timezone


@user_bp.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('user.dashboard'))
    return redirect(url_for('auth.login'))


@user_bp.route('/dashboard')
@login_required
def dashboard():
    username = session.get('username')
    return render_template('dashboard.html', username=username)


@user_bp.route('/parking-session')
@login_required
def parking_session():
    """Render parking session retrieval page"""
    username = session.get('username')
    return render_template('parking_session.html', username=username)


@user_bp.route('/parking-session/lookup', methods=['POST'])
@login_required
def parking_session_lookup():
    """Lookup parking session by plate number and delete after retrieval"""
    try:
        data = request.get_json()
        plate_number = data.get('plate_number', '').strip()

        if not plate_number:
            return jsonify({
                'success': False,
                'message': 'Plate number is required'
            }), 400

        # Hash the plate number
        plate_hash = hash_plate_number(plate_number)

        # Query database
        parking_session = ParkingSession.query.filter_by(car_id_hash=plate_hash).first()

        if not parking_session:
            return jsonify({
                'success': False,
                'message': 'No parking session found for this plate number'
            }), 404

        # Extract session data
        timestamp_start = parking_session.timestamp_start
        timestamp_end = parking_session.timestamp_end

        # Check if session is still active (no end time)
        is_active = timestamp_start and timestamp_end is None

        # Calculate duration
        if timestamp_start and timestamp_end:
            # Completed session
            duration_minutes = calculate_duration(
                timestamp_start.isoformat(),
                timestamp_end.isoformat()
            )
        elif is_active:
            # Active session - calculate current duration
            from datetime import datetime, timezone
            duration_minutes = calculate_duration(
                timestamp_start.isoformat(),
                datetime.now(timezone.utc).isoformat()
            )
        else:
            duration_minutes = 0

        # Prepare response data
        session_data = {
            'plate_number': plate_number.upper(),
            'timestamp_start': timestamp_start.isoformat() if timestamp_start else None,
            'timestamp_end': timestamp_end.isoformat() if timestamp_end else None,
            'duration_minutes': duration_minutes,
            'is_active': is_active
        }

        # Only delete if session is completed (has both start and end time)
        if timestamp_start and timestamp_end:
            db.session.delete(parking_session)
            db.session.commit()
            message = 'Parking session retrieved successfully. Record has been deleted.'
        else:
            # Don't delete active sessions
            message = 'Active parking session retrieved. Your current parking duration is shown. Session will remain active until you exit.'

        return jsonify({
            'success': True,
            'message': message,
            'data': session_data
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500
