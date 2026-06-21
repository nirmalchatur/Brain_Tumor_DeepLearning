from flask import Flask, render_template, request, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import numpy as np
import re
import smtplib
import os
import cv2
import random 
import uuid
from pathlib import Path
import logging

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.logger.setLevel(logging.DEBUG)

# Configure upload folders
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/outputs/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# MongoDB Setup
try:
    client = MongoClient(MONGO_DB_URL, 
                        connectTimeoutMS=30000, socketTimeoutMS=None)
    db = client['podcast']
    users_collection = db['users']
    predictions_collection = db['predictions']
    print("Successfully connected to MongoDB")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    raise SystemExit(1)
def load_users():
    try:
        users = users_collection.find({}, {'username': 1, 'password': 1, '_id': 0})
        return {user['username']: user['password'] for user in users if 'username' in user and 'password' in user}
    except Exception as e:
        print(f"Error loading users from MongoDB: {e}")
        return {}
def send_otp(to_email, otp):
    sender_email = "nirmal.chaturvedi@mitaoe.ac.in"
    sender_password = "qlzj gacb abfz proc"
    subject = "Your OTP for Brain Tumor Detection Registration"
    body = f"Your OTP is: {otp}"
    email_text = f"Subject: {subject}\n\n{body}"
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, email_text)
        print(f"OTP sent to {to_email}")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# Load Models
model_paths = {
    'ResNet50': "model/Brain_Tumor_ResNet50_CNN_Improved.h5",
    'InceptionV3': "model/model_update.h5",
    'VGG16':"model\Brain_Tumor_VGG16_CNN.h5"
}

loaded_models = {}
for name, path in model_paths.items():
    try:
        loaded_models[name] = load_model(path)
        app.logger.info(f"{name} loaded successfully from {path}")
    except Exception as e:
        app.logger.error(f"Error loading {name}: {e}")
        raise SystemExit(1)

# Constants
IMG_SIZE = (224, 224)
LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def annotate_image(img_path, tumor_type, output_path):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError("Could not read image")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f'Tumor Type: {tumor_type}', (10, 30), font, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(output_path), img)
        return True
    except Exception as e:
        app.logger.error(f"Error annotating image: {e}")
        return False
pending_users = {}
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form['username'].strip()
            email = request.form['email'].strip()
            password = request.form['password']

            # Basic server-side validation
            if not username or not email or not password:
                return render_template('register.html', error='All fields are required.')

            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                return render_template('register.html', error='Invalid email address.')

            if len(password) < 8:
                return render_template('register.html', error='Password must be at least 8 characters long.')

            existing_user = users_collection.find_one({'email': email})
            if existing_user:
                return render_template('register.html', error='Email already registered.')

            # Generate OTP
            otp = str(random.randint(100000, 999999))
            pending_users[email] = {
                'username': username,
                'email': email,
                'password': password,  # Save password in plain text
                'otp': otp
            }

            # Send OTP
            if send_otp(email, otp):
                return render_template('verify.html', email=email)
            else:
                return render_template('register.html', error='Failed to send OTP. Try again later.')
        except Exception as e:
            print(f"[REGISTER ERROR] {e}")
            return render_template('register.html', error='An error occurred during registration.')

    return render_template('register.html')

     
@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    """
    Route to handle OTP verification.
    After successful verification, the user is redirected to the login page.
    """
    try:
        email = request.form['email']  # Get the email from the form
        entered_otp = request.form['otp']  # Get the entered OTP from the form
        
        # Check if the email exists in the pending_users dictionary
        if email in pending_users:
            stored_otp = pending_users[email]['otp']  # Get the OTP stored for the user

            if entered_otp == stored_otp:
                # If OTP is correct, register the user and insert into the MongoDB collection
                user_data = pending_users[email]
                
                # Insert or update the user in the collection (MongoDB)
                users_collection.update_one(
                    {'email': email},  # Search by email
                    {'$set': user_data},  # Set the new user data
                    upsert=True  # Create a new document if the user doesn't exist
                )
                
                del pending_users[email]  # Remove the user from pending_users after successful verification
                return redirect(url_for('login'))  # Redirect to the login page after successful OTP verification
            else:
                # If OTP is incorrect, render the OTP page again with an error message
                return render_template('verify.html', email=email, error="Invalid OTP. Please try again.")
        else:
            # If email is not found in the pending_users dictionary, show an error
            return render_template('verify.html', error="No OTP found for this email.")
    
    except KeyError as e:
        # Handle missing 'email' or 'otp' fields in the form submission
        return render_template('verify.html', error="Missing fields. Please check your input.")

def predict_tumor_type(img_path, model_name):
    model = loaded_models.get(model_name)
    if not model:
        app.logger.error(f"Model {model_name} not loaded")
        return "Model not loaded"
    
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        predictions = model.predict(img_array)
        return LABELS[np.argmax(predictions)]
    except Exception as e:
        app.logger.error(f"Error predicting tumor type: {e}")
        return "Prediction Error"

@app.route('/', methods=['GET'])
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
     # Get accuracy data (replace with your actual accuracy calculation)
    accuracy_data = {
        'ResNet50': 0.957,  # 95.7%
        'InceptionV3': 0.943,  # 94.3%
        'VGG16': 0.925  # 92.5%
    }
    return render_template('index.html',
                         model_choices=list(model_paths.keys()),
                         model_used=session.get('model_used', 'ResNet50'),
                         accuracy_data=accuracy_data)  # Make sure this is included


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # Add your authentication logic here
        session['username'] = username
        return redirect(url_for('index'))
    return render_template('login.html')
@app.route('/logout')
def logout():
    # Clear the user session completely
    session.clear()
    # Redirect to login page with a success message
    return redirect(url_for('login'))
@app.route('/get-accuracy')
def get_accuracy():
    # Replace with your actual accuracy calculation logic
    accuracy_data = {
        'ResNet50': 0.957,
        'InceptionV3': 0.943,
        'VGG16': 0.925
    }
    return jsonify(accuracy_data)
@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    try:
        # Get model selection
        model_name = request.form.get('model_choice', 'ResNet50')
        app.logger.debug(f"Model selected: {model_name}")
        
        if model_name not in loaded_models:
            flash("Selected model not available", "error")
            return redirect(url_for('index'))
        
        # Handle file upload
        if 'file' not in request.files:
            flash("No file uploaded", "error")
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash("No file selected", "error")
            return redirect(url_for('index'))
        
        # Validate file
        filename = secure_filename(file.filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            flash("Invalid file type. Please upload an image.", "error")
            return redirect(url_for('index'))
        
        # Save file
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{unique_id}_{filename}"
        input_path = Path(app.config['UPLOAD_FOLDER']) / filename
        output_path = Path(app.config['OUTPUT_FOLDER']) / filename
        
        file.save(input_path)
        app.logger.debug(f"File saved to: {input_path}")
        
        # Make prediction
        tumor_type = predict_tumor_type(input_path, model_name)
        if tumor_type == "Prediction Error":
            flash("Error processing image", "error")
            return redirect(url_for('index'))
        
        # Annotate image
        if not annotate_image(input_path, tumor_type, output_path):
            flash("Error generating visualization", "error")
            return redirect(url_for('index'))
        
        # Store model used in session
        session['model_used'] = model_name
        
        # Prepare results
        prediction = f"{tumor_type} Tumor Detected" if tumor_type != "No Tumor" else "No Tumor Detected"
        
        return render_template('index.html',
                            prediction=prediction,
                            accuracy=95.7,
                            original_filename=filename,
                            output_filename=filename,
                            model_choices=list(model_paths.keys()),
                            model_used=model_name)
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}", exc_info=True)
        flash("An error occurred during processing", "error")
        return redirect(url_for('index'))
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
