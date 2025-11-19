from . import auth_bp
from server.parkingpal.extensions import db
from flask import render_template, request, redirect, url_for, session, flash
from server.parkingpal.utils import hash_password
from sqlalchemy.exc import IntegrityError
from ...models import User


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return redirect(url_for('auth.register'))

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('auth.register'))

        if len(password) < 6:
            flash('Password must be at least 6 characters long!', 'error')
            return redirect(url_for('auth.register'))

        hashed_password = hash_password(password)
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)

        try:
            db.session.commit()

            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('auth.login'))

        except IntegrityError:
            db.session.rollback()
            flash('Username or email already exists!', 'error')
            return redirect(url_for('auth.register'))

    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash('Please enter both username and password!', 'error')
            return redirect(url_for('login'))

        hashed_password = hash_password(password)

        user = User.query.filter_by(username=username, password=hashed_password).first()

        if user:
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('user.dashboard'))
        else:
            flash('Invalid username or password!', 'error')
            return redirect(url_for('auth.login'))

    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    username = session.get('username')
    session.clear()
    flash(f'Goodbye, {username}! You have been logged out.', 'info')
    return redirect(url_for('auth.login'))
