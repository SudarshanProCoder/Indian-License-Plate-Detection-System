from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from plate_detector import LicensePlateDetector
from state_codes import STATE_CODES
from datetime import datetime
import traceback
from database import Database

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize license plate detector and database
detector = LicensePlateDetector()
db = Database()

def generate_frames():
    while True:
        frame_data = detector.get_frame()
        if frame_data is None:
            break
            
        frame, text, error = frame_data
        if frame is not None:
            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def get_state_from_plate(plate_number):
    """Extract state code from plate number and return state name"""
    if len(plate_number) >= 2:
        state_code = plate_number[:2]
        return STATE_CODES.get(state_code, "Unknown State")
    return "Unknown State"

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/detect')
def detect_page():
    return render_template('index.html')

@app.route('/history')
def history():
    """Display history of detected license plates"""
    plates = db.get_all_plates()
    return render_template('history.html', plates=plates)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    success = detector.start_camera()
    return jsonify({'success': success})

@app.route('/stop_camera')
def stop_camera():
    detector.stop_camera()
    return jsonify({'success': True})

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded. Please select an image file.'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected. Please choose an image file.'})
        
        # Check file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'})
        
        # Save the uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'plate_{timestamp}.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Verify the image was saved correctly
        if not os.path.exists(filepath):
            return jsonify({'error': 'Failed to save the uploaded image. Please try again.'})
        
        # Read and verify the image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Failed to read the image. The file might be corrupted.'})
        
        # Check image dimensions
        height, width = img.shape[:2]
        if width < 100 or height < 100:
            return jsonify({'error': 'Image is too small. Please upload a larger image.'})
        
        # Detect license plate
        text, plate_img, error = detector.detect_plate(filepath)
        
        if error:
            return jsonify({'error': f'Detection error: {error}'})
        
        if text:
            # Get state information
            state = get_state_from_plate(text)
            
            # Save the processed image
            processed_filename = f'processed_{filename}'
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_filepath, plate_img)
            
            # Store in database
            db.add_plate(text, state, processed_filepath)
            
            return jsonify({
                'plate_number': text,
                'state': state,
                'image_url': f'/uploads/{processed_filename}',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return jsonify({'error': 'No license plate detected in the image. Please ensure the license plate is clearly visible.'})
    
    except Exception as e:
        # Log the full error for debugging
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True) 