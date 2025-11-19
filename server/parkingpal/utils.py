from datetime import datetime
import hashlib
from flask import flash, session, redirect, url_for
from functools import wraps
from .models.user import User


def calculate_duration(entrance_time, exit_time):
    try:
        start = datetime.fromisoformat(entrance_time)
        end = datetime.fromisoformat(exit_time)
        duration = (end - start).total_seconds() / 60
        return int(duration)
    except:
        return 0


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('auth.login'))

        user = session.get('user_id')
        if user is None:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('auth.login'))

        is_admin = User.query.filter_by(id=user).first().is_admin

        if not is_admin:
            flash('You do not have access to this page. Forbidden', 'warning')
            return redirect(url_for('auth.login'))

        return f(*args, **kwargs)

    return decorated_function
