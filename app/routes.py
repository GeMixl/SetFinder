# app/routes.py
from flask import render_template, current_app as app
from flask import request, jsonify
import base64
import os
from .backend.setfinder.setfinder import find_set_from_deck  # Import the function from setfinder.py

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    photo_data = request.form['photo']
    photo_data = photo_data.replace('data:image/png;base64,', '')
    photo_data = base64.b64decode(photo_data)
    photo_path = os.path.join('uploads', 'raw_image.png')
    with open(photo_path, 'wb') as photo_file:
        photo_file.write(photo_data)
    return jsonify({'message': 'Photo uploaded successfully!'})

@app.route('/find_sets', methods=['GET'])
def find_sets():
    sets_found_in_upload = find_set_from_deck()
    return jsonify(sets_found_in_upload)

