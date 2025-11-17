from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import hashlib
import os
from functools import wraps
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

DATABASE = 'parkingpal.db'


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_admin BOOLEAN NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS cars (
            car_id_hash TEXT PRIMARY KEY,
            timestamp_start TIMESTAMP,
            timestamp_end TIMESTAMP,
            user_id INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS parking_spaces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            space_number INTEGER UNIQUE NOT NULL,
            is_free BOOLEAN NOT NULL DEFAULT 1,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS parking_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            space_number INTEGER NOT NULL,
            status TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (space_number) REFERENCES parking_spaces(space_number)
        );
    ''')
    conn.commit()
    print("Database initialized")
    admin = conn.execute("SELECT * FROM users WHERE username = 'admin'")
    if not admin.fetchall():
        conn.execute("INSERT INTO users (username, email, password, is_admin) VALUES (?, ?, ?, ?)", ('admin', "admin@admin.no", hash_password("Password1."),  1))
        conn.commit()
        conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))

        user = session.get('user_id')
        if user is None:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))

        conn = get_db_connection()
        is_admin = conn.execute("SELECT is_admin FROM users where id = ?", (user,)).fetchone()
        conn.commit()
        conn.close()

        if not is_admin:
            flash('You do not have access to this page. Forbidden', 'warning')
            return redirect(url_for('login'))

        return f(*args, **kwargs)
    return decorated_function




# ============= API ENDPOINTS FOR CAMERA =============

@app.route('/api/parking/update', methods=['POST'])
def update_parking_spaces():
    """
    Endpoint for camera to send parking space updates
    Expected JSON: {
        "spaces": [
            {"space_number": 1, "is_free": true},
            {"space_number": 2, "is_free": false},
            ...
        ]
    }
    """
    try:
        data = request.get_json()

        if not data or 'spaces' not in data:
            return jsonify({'error': 'Invalid data format'}), 400

        conn = get_db_connection()

        for space in data['spaces']:
            space_number = space.get('space_number')
            is_free = space.get('is_free')

            if space_number is None or is_free is None:
                continue

            # Check if space exists
            existing = conn.execute(
                'SELECT * FROM parking_spaces WHERE space_number = ?',
                (space_number,)
            ).fetchone()

            if existing:
                # Update existing space
                conn.execute(
                    'UPDATE parking_spaces SET is_free = ?, last_updated = ? WHERE space_number = ?',
                    (is_free, datetime.now(), space_number)
                )
            else:
                # Insert new space
                conn.execute(
                    'INSERT INTO parking_spaces (space_number, is_free, last_updated) VALUES (?, ?, ?)',
                    (space_number, is_free, datetime.now())
                )

            # Log to history
            status = 'free' if is_free else 'occupied'
            conn.execute(
                'INSERT INTO parking_history (space_number, status, timestamp) VALUES (?, ?, ?)',
                (space_number, status, datetime.now())
            )

        conn.commit()
        conn.close()

        return jsonify({'success': True, 'message': 'Parking spaces updated'}), 200

    except Exception as e:
        print(f"Error updating parking spaces: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/parking/status', methods=['GET'])
def get_parking_status():
    """
    Get current parking status
    Returns: {
        "total": 12,
        "available": 5,
        "occupied": 7,
        "spaces": [...]
    }
    """
    try:
        conn = get_db_connection()
        spaces = conn.execute(
            'SELECT space_number, is_free, last_updated FROM parking_spaces ORDER BY space_number'
        ).fetchall()
        conn.close()

        spaces_list = [
            {
                'id': space['space_number'],
                'status': 'available' if space['is_free'] else 'occupied',
                'last_updated': space['last_updated']
            }
            for space in spaces
        ]

        total = len(spaces_list)
        available = sum(1 for s in spaces_list if s['status'] == 'available')
        occupied = total - available

        return jsonify({
            'total': total,
            'available': available,
            'occupied': occupied,
            'spaces': spaces_list
        }), 200

    except Exception as e:
        print(f"Error getting parking status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/parking/history', methods=['GET'])
def get_parking_history():
    """
    Get parking history for analytics
    Optional query params: space_number, limit
    """
    try:
        space_number = request.args.get('space_number', type=int)
        limit = request.args.get('limit', default=100, type=int)

        conn = get_db_connection()

        if space_number:
            history = conn.execute(
                'SELECT * FROM parking_history WHERE space_number = ? ORDER BY timestamp DESC LIMIT ?',
                (space_number, limit)
            ).fetchall()
        else:
            history = conn.execute(
                'SELECT * FROM parking_history ORDER BY timestamp DESC LIMIT ?',
                (limit,)
            ).fetchall()

        conn.close()

        history_list = [
            {
                'space_number': h['space_number'],
                'status': h['status'],
                'timestamp': h['timestamp']
            }
            for h in history
        ]

        return jsonify({'history': history_list}), 200

    except Exception as e:
        print(f"Error getting parking history: {e}")
        return jsonify({'error': str(e)}), 500


# ============= WEB ROUTES =============

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('register'))

        if len(password) < 6:
            flash('Password must be at least 6 characters long!', 'error')
            return redirect(url_for('register'))

        hashed_password = hash_password(password)

        try:
            conn = get_db_connection()
            conn.execute(
                'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                (username, email, hashed_password)
            )
            conn.commit()
            conn.close()

            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
            return redirect(url_for('register'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash('Please enter both username and password!', 'error')
            return redirect(url_for('login'))

        hashed_password = hash_password(password)

        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ? AND password = ?',
            (username, hashed_password)
        ).fetchone()
        conn.close()

        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f'Welcome back, {user["username"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/dashboard')
@login_required
def dashboard():
    username = session.get('username')
    return render_template('dashboard.html', username=username)


@app.route('/logout')
def logout():
    username = session.get('username')
    session.clear()
    flash(f'Goodbye, {username}! You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/admin')
@admin_required
def admin():
    username = session.get('username')
    return render_template('admin.html', username=username)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)