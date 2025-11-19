from . import user_bp
from server.parkingpal.utils import login_required
from flask import redirect, url_for, session, render_template

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