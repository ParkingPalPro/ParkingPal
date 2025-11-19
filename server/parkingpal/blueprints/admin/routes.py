from . import admin_bp
from server.parkingpal.utils import admin_required
from flask import render_template, session


@admin_bp.route('/admin')
@admin_required
def admin():
    username = session.get('username')
    return render_template('admin.html', username=username)
