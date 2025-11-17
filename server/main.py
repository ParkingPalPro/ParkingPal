from flask import Flask, request, jsonify
from datetime import datetime
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading

app = Flask(__name__)

# Email Configuration - UPDATE THESE!
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "your_email@gmail.com"
SMTP_PASSWORD = "your_app_password"  # Use App Password for Gmail
FROM_EMAIL = "parking@yourdomain.com"


# Database initialization
def init_db():
    conn = sqlite3.connect('parking.db')
    c = conn.cursor()

    # Parking sessions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS parking_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hashed_plate TEXT NOT NULL,
            original_plate TEXT,
            user_email TEXT,
            entrance_time TEXT NOT NULL,
            exit_time TEXT,
            entrance_camera TEXT,
            exit_camera TEXT,
            status TEXT DEFAULT 'active'
        )
    ''')

    # Monitoring events table
    c.execute('''
        CREATE TABLE IF NOT EXISTS monitoring_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hashed_plate TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            event_type TEXT
        )
    ''')

    # User plate associations (for email lookup)
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_plates (
            hashed_plate TEXT PRIMARY KEY,
            original_plate TEXT NOT NULL,
            user_email TEXT NOT NULL,
            registered_date TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()


def get_db():
    conn = sqlite3.connect('parking.db')
    conn.row_factory = sqlite3.Row
    return conn


def send_parking_receipt(user_email, plate_number, entrance_time, exit_time, duration_minutes):
    """Send parking receipt via email"""
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'Parking Receipt - {plate_number}'
        msg['From'] = FROM_EMAIL
        msg['To'] = user_email

        # Create HTML email body
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: #4CAF50; color: white; padding: 20px; text-align: center;">
                <h1>Parking Receipt</h1>
            </div>
            <div style="padding: 20px; background-color: #f9f9f9;">
                <h2>Thank you for parking with us!</h2>

                <div style="background-color: white; padding: 15px; margin: 10px 0; border-radius: 5px;">
                    <p><strong>Vehicle:</strong> {plate_number}</p>
                    <p><strong>Entry Time:</strong> {entrance_time}</p>
                    <p><strong>Exit Time:</strong> {exit_time}</p>
                    <p><strong>Duration:</strong> {duration_minutes} minutes ({duration_minutes // 60}h {duration_minutes % 60}m)</p>
                </div>

                <p style="color: #666; font-size: 12px; margin-top: 20px;">
                    This is an automated message. Please do not reply to this email.
                </p>
            </div>
        </body>
        </html>
        """

        part = MIMEText(html, 'html')
        msg.attach(part)

        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        print(f"✓ Email sent to {user_email}")
        return True

    except Exception as e:
        print(f"✗ Email error: {e}")
        return False


def calculate_duration(entrance_time, exit_time):
    """Calculate parking duration in minutes"""
    try:
        start = datetime.fromisoformat(entrance_time)
        end = datetime.fromisoformat(exit_time)
        duration = (end - start).total_seconds() / 60
        return int(duration)
    except:
        return 0


@app.route('/plate_event', methods=['POST'])
def handle_plate_event():
    """Handle plate detection events from cameras"""
    data = request.json

    camera_id = data.get('camera_id')
    role = data.get('role')
    hashed_plate = data.get('hashed_plate')
    timestamp = data.get('timestamp')
    original_plate = data.get('original_plate')

    if not all([camera_id, role, hashed_plate, timestamp]):
        return jsonify({"error": "Missing required fields"}), 400

    conn = get_db()
    c = conn.cursor()

    try:
        if role == 'entrance':
            # Check if there's already an active session
            c.execute('''
                SELECT id FROM parking_sessions 
                WHERE hashed_plate = ? AND status = 'active'
            ''', (hashed_plate,))

            existing = c.fetchone()

            if existing:
                return jsonify({
                    "status": "warning",
                    "message": "Vehicle already has active parking session"
                }), 200

            # Get or create user email association
            user_email = None
            if original_plate:
                c.execute('SELECT user_email FROM user_plates WHERE hashed_plate = ?',
                          (hashed_plate,))
                user_row = c.fetchone()

                if user_row:
                    user_email = user_row['user_email']
                else:
                    # For demo: use placeholder email. In production, prompt for email
                    user_email = f"user_{original_plate}@example.com"
                    c.execute('''
                        INSERT OR REPLACE INTO user_plates 
                        (hashed_plate, original_plate, user_email, registered_date)
                        VALUES (?, ?, ?, ?)
                    ''', (hashed_plate, original_plate, user_email, timestamp))

            # Create new parking session
            c.execute('''
                INSERT INTO parking_sessions 
                (hashed_plate, original_plate, user_email, entrance_time, entrance_camera, status)
                VALUES (?, ?, ?, ?, ?, 'active')
            ''', (hashed_plate, original_plate, user_email, timestamp, camera_id))

            conn.commit()

            return jsonify({
                "status": "success",
                "message": f"Vehicle entered - Session started",
                "session_id": c.lastrowid
            }), 200

        elif role == 'exit':
            # Find active session
            c.execute('''
                SELECT id, original_plate, user_email, entrance_time 
                FROM parking_sessions 
                WHERE hashed_plate = ? AND status = 'active'
            ''', (hashed_plate,))

            session = c.fetchone()

            if not session:
                return jsonify({
                    "status": "warning",
                    "message": "No active parking session found"
                }), 200

            # Update session with exit info
            c.execute('''
                UPDATE parking_sessions 
                SET exit_time = ?, exit_camera = ?, status = 'completed'
                WHERE id = ?
            ''', (timestamp, camera_id, session['id']))

            conn.commit()

            # Calculate duration and send email in background
            duration = calculate_duration(session['entrance_time'], timestamp)

            if session['user_email']:
                # Send email in separate thread to not block response
                threading.Thread(
                    target=send_parking_receipt,
                    args=(
                        session['user_email'],
                        session['original_plate'] or hashed_plate[:8],
                        session['entrance_time'],
                        timestamp,
                        duration
                    )
                ).start()

            return jsonify({
                "status": "success",
                "message": f"Vehicle exited - Duration: {duration} min",
                "duration_minutes": duration
            }), 200

        elif role == 'monitoring':
            # Log monitoring event
            c.execute('''
                INSERT INTO monitoring_events 
                (hashed_plate, camera_id, timestamp, event_type)
                VALUES (?, ?, ?, 'detected')
            ''', (hashed_plate, camera_id, timestamp))

            conn.commit()

            # Check if vehicle has active session
            c.execute('''
                SELECT entrance_time FROM parking_sessions 
                WHERE hashed_plate = ? AND status = 'active'
            ''', (hashed_plate,))

            session = c.fetchone()
            status = "Active session" if session else "No active session"

            return jsonify({
                "status": "success",
                "message": f"Monitoring event logged - {status}"
            }), 200

        else:
            return jsonify({"error": "Invalid camera role"}), 400

    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route('/register_plate', methods=['POST'])
def register_plate():
    """Register a plate with user email"""
    data = request.json

    hashed_plate = data.get('hashed_plate')
    original_plate = data.get('original_plate')
    user_email = data.get('user_email')

    if not all([hashed_plate, original_plate, user_email]):
        return jsonify({"error": "Missing required fields"}), 400

    conn = get_db()
    c = conn.cursor()

    try:
        c.execute('''
            INSERT OR REPLACE INTO user_plates 
            (hashed_plate, original_plate, user_email, registered_date)
            VALUES (?, ?, ?, ?)
        ''', (hashed_plate, original_plate, user_email, datetime.now().isoformat()))

        conn.commit()
        return jsonify({"status": "success", "message": "Plate registered"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route('/active_sessions', methods=['GET'])
def get_active_sessions():
    """Get all active parking sessions"""
    conn = get_db()
    c = conn.cursor()

    c.execute('''
        SELECT id, original_plate, entrance_time, entrance_camera 
        FROM parking_sessions 
        WHERE status = 'active'
        ORDER BY entrance_time DESC
    ''')

    sessions = [dict(row) for row in c.fetchall()]
    conn.close()

    return jsonify({"active_sessions": sessions}), 200


@app.route('/session_history', methods=['GET'])
def get_session_history():
    """Get parking session history"""
    conn = get_db()
    c = conn.cursor()

    limit = request.args.get('limit', 50, type=int)

    c.execute('''
        SELECT original_plate, entrance_time, exit_time, 
               entrance_camera, exit_camera, status
        FROM parking_sessions 
        ORDER BY entrance_time DESC
        LIMIT ?
    ''', (limit,))

    sessions = [dict(row) for row in c.fetchall()]
    conn.close()

    return jsonify({"sessions": sessions}), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }), 200


if __name__ == '__main__':
    print("Initializing Parking Management Server...")
    init_db()
    print("Database initialized")
    print("\nServer Configuration:")
    print(f"  SMTP Server: {SMTP_SERVER}")
    print(f"  From Email: {FROM_EMAIL}")
    print("\nEndpoints:")
    print("  POST /plate_event - Handle camera events")
    print("  POST /register_plate - Register plate with email")
    print("  GET /active_sessions - Get active parking sessions")
    print("  GET /session_history - Get session history")
    print("  GET /health - Health check")
    print("\nStarting server on http://0.0.0.0:5000")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)
