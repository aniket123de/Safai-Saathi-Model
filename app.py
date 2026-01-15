import cv2
from ultralytics import YOLO
import argparse
from flask import Flask, Response, render_template, request, jsonify
import tempfile
import os
import requests
import json
from datetime import datetime


# Flask APPLICATION
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# ============================================================================
# FIREBASE DASHBOARD CONFIGURATION
# ============================================================================

# Your dashboard URL (update when deployed)
DASHBOARD_URL = "http://localhost:3000"  # Local development
# DASHBOARD_URL = "https://your-dashboard.vercel.app"  # Production

# API endpoint
UPLOAD_ENDPOINT = f"{DASHBOARD_URL}/api/model/upload"

# Optional: Add API key for production security
API_KEY = None  # Set to your API key if you add authentication

# Load the YOLOv8 model (using the trained weights)
model = YOLO('datasets/garbage-can-overflow-4/runs/detect/train2/weights/best.pt')

# Flag to indicate if the script should terminate
terminate_flag = False

# Store detection data with location
detection_logs = []
gps_location = None  # Store the latest GPS location

# Geolocation function using a free IP geolocation service
def get_location_from_ip(ip_address):
    try:
        # If we have a localhost IP, try to get public IP first
        if ip_address in ['127.0.0.1', 'localhost', '::1']:
            try:
                response = requests.get('https://api.ipify.org?format=json', timeout=5)
                if response.status_code == 200:
                    ip_address = response.json().get('ip')
                    print(f"Got public IP: {ip_address}")
            except:
                # Try alternative service
                try:
                    response = requests.get('https://httpbin.org/ip', timeout=5)
                    if response.status_code == 200:
                        ip_address = response.json().get('origin', '').split(',')[0].strip()
                        print(f"Got public IP from httpbin: {ip_address}")
                except:
                    pass
        
        # Now get location data
        response = requests.get(f'https://ipapi.co/{ip_address}/json/', timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'source': 'IP',
                'ip': ip_address,
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'country': data.get('country_name', 'Unknown'),
                'latitude': data.get('latitude'),
                'longitude': data.get('longitude'),
                'accuracy': 'City-level (~10km)',
                'timestamp': datetime.now().isoformat()
            }
    except Exception as e:
        print(f"Error getting location from IP: {e}")
    
    return None

# Reverse geocoding function to get address from coordinates
def get_address_from_coords(lat, lng):
    try:
        # Using Nominatim (OpenStreetMap) for reverse geocoding
        response = requests.get(
            f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}',
            headers={'User-Agent': 'GarbageDetectionApp/1.0'},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data.get('display_name', 'Address not found')
    except Exception as e:
        print(f"Error getting address: {e}")
    return None

# Get client IP address
def get_client_ip():
    # Check various headers for real IP
    possible_headers = [
        'HTTP_X_FORWARDED_FOR',
        'HTTP_X_REAL_IP',
        'HTTP_X_FORWARDED',
        'HTTP_X_CLUSTER_CLIENT_IP',
        'HTTP_FORWARDED_FOR',
        'HTTP_FORWARDED',
        'REMOTE_ADDR'
    ]
    
    for header in possible_headers:
        ip = request.environ.get(header)
        if ip and ip != '127.0.0.1' and ip != 'localhost':
            # Handle comma-separated IPs (X-Forwarded-For can have multiple IPs)
            if ',' in ip:
                ip = ip.split(',')[0].strip()
            return ip
    
    # If we still have localhost, try to get public IP
    try:
        # Use a service to get the public IP
        response = requests.get('https://httpbin.org/ip', timeout=5)
        if response.status_code == 200:
            public_ip = response.json().get('origin', '').split(',')[0].strip()
            if public_ip and public_ip != '127.0.0.1':
                return public_ip
    except:
        pass
    
    # Fallback to environment IP
    return request.environ.get('REMOTE_ADDR', '127.0.0.1')

# ============================================================================
# FIREBASE UPLOAD FUNCTIONS
# ============================================================================

def upload_detection_to_firebase(detection_count, confidence_scores, location_data):
    """
    Upload garbage detection to Firebase via dashboard API
    
    Args:
        detection_count (int): Number of garbage items detected
        confidence_scores (list): List of confidence scores (0-1)
        location_data (dict): Location information with GPS/IP data
    
    Returns:
        bool: True if upload successful, False otherwise
    """
    try:
        # Prepare payload matching the dashboard API format
        payload = {
            "detection_count": detection_count,
            "confidence_scores": confidence_scores,
            "location": {
                "source": location_data.get('source', 'IP'),
                "latitude": location_data.get('latitude', 0),
                "longitude": location_data.get('longitude', 0),
                "accuracy": location_data.get('accuracy', 'Unknown'),
                "address": location_data.get('address', 'Unknown'),
                "timestamp": location_data.get('timestamp', datetime.now().isoformat())
            },
            "timestamp": datetime.now().isoformat(),
            "model_version": "YOLOv8"
        }
        
        # Add optional fields if available
        if location_data.get('city'):
            payload['location']['city'] = location_data['city']
        if location_data.get('region'):
            payload['location']['region'] = location_data['region']
        if location_data.get('country'):
            payload['location']['country'] = location_data['country']
        if location_data.get('ip'):
            payload['location']['ip'] = location_data['ip']
        
        # Prepare headers
        headers = {'Content-Type': 'application/json'}
        if API_KEY:
            headers['Authorization'] = f'Bearer {API_KEY}'
        
        # Send POST request to dashboard
        response = requests.post(
            UPLOAD_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=10
        )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Successfully uploaded to Firebase: {result.get('documentId', 'N/A')}")
            return True
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error uploading to Firebase: {e}")
        return False
    except Exception as e:
        print(f"❌ Error uploading to Firebase: {e}")
        return False


def batch_upload_detections_to_firebase(detection_logs_list):
    """
    Batch upload multiple detections to Firebase
    
    Args:
        detection_logs_list (list): List of detection dictionaries
    
    Returns:
        dict: Upload statistics
    """
    try:
        # Prepare batch payload
        batch_payload = []
        
        for log in detection_logs_list:
            payload = {
                "detection_count": log.get('detection_count', 0),
                "confidence_scores": log.get('confidence_scores', []),
                "location": {
                    "source": log['location'].get('source', 'IP'),
                    "latitude": log['location'].get('latitude', 0),
                    "longitude": log['location'].get('longitude', 0),
                    "accuracy": log['location'].get('accuracy', 'Unknown'),
                    "address": log['location'].get('address', 'Unknown'),
                    "timestamp": log['location'].get('timestamp', datetime.now().isoformat())
                },
                "timestamp": log.get('timestamp', datetime.now().isoformat()),
                "model_version": "YOLOv8"
            }
            
            # Add optional fields
            if log['location'].get('city'):
                payload['location']['city'] = log['location']['city']
            if log['location'].get('region'):
                payload['location']['region'] = log['location']['region']
            if log['location'].get('country'):
                payload['location']['country'] = log['location']['country']
            if log['location'].get('ip'):
                payload['location']['ip'] = log['location']['ip']
            
            batch_payload.append(payload)
        
        # Send batch request
        headers = {'Content-Type': 'application/json'}
        if API_KEY:
            headers['Authorization'] = f'Bearer {API_KEY}'
        
        response = requests.post(
            UPLOAD_ENDPOINT,
            json=batch_payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch upload successful: {result.get('message', 'N/A')}")
            return result.get('details', {})
        else:
            print(f"❌ Batch upload failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error in batch upload: {e}")
        return None

# Store detection data with location
def log_detection_with_location(detection_count, confidence_scores, client_ip=None):
    # Try to use GPS location first, fallback to IP location
    location_data = None
    
    if gps_location:
        location_data = gps_location.copy()
        location_data['source'] = 'GPS'
        print(f"Using GPS location: {location_data['latitude']}, {location_data['longitude']}")
    else:
        # Use provided IP or try to get it from request context if available
        if client_ip is None:
            try:
                client_ip = get_client_ip()
            except RuntimeError:
                # If outside request context, use a default IP
                client_ip = '127.0.0.1'
        location_data = get_location_from_ip(client_ip)
        if location_data:
            print(f"Using IP location: {client_ip}")
    
    if location_data:
        log_entry = {
            'detection_count': detection_count,
            'confidence_scores': confidence_scores,
            'location': location_data,
            'timestamp': datetime.now().isoformat()
        }
        detection_logs.append(log_entry)
        
        location_str = f"{location_data.get('city', 'Unknown')}, {location_data.get('country', 'Unknown')}"
        if location_data.get('source') == 'GPS':
            location_str = f"GPS ({location_data['latitude']:.4f}, {location_data['longitude']:.4f})"
        
        print(f"Detection logged: {detection_count} items at {location_str}")
        
        # Upload to Firebase dashboard
        upload_detection_to_firebase(detection_count, confidence_scores, location_data)
        
        # Keep only the last 100 logs to prevent memory issues
        if len(detection_logs) > 100:
            detection_logs.pop(0)

# Define a generator function to stream video frames to the web page
def generate(file_path):
    if file_path == "camera":
        cap = cv2.VideoCapture(0)
    elif file_path == "ngrok":
        # Use the ngrok URL for mobile phone CCTV
        # Try different possible video stream endpoints
        possible_urls = [
            "https://76ee592fbf21.ngrok-free.app/video",
        ]
        
        cap = None
        for url in possible_urls:
            print(f"Trying to connect to: {url}")
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                print(f"Successfully connected to: {url}")
                break
            else:
                print(f"Failed to connect to: {url}")
                cap.release()
        
        if not cap or not cap.isOpened():
            print("Could not connect to any ngrok URL")
            return
    else:
        cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        # Read a frame from the video file
        success, frame = cap.read()
        
        print(f"Frame read success: {success}")  # Debug output

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Extract detection information
            detections = results[0].boxes
            if detections is not None and len(detections) > 0:
                detection_count = len(detections)
                confidence_scores = detections.conf.tolist() if detections.conf is not None else []
                
                # Log detection with geolocation (every 30 frames to avoid spam)
                if hasattr(generate, 'frame_count'):
                    generate.frame_count += 1
                else:
                    generate.frame_count = 1
                
                if generate.frame_count % 30 == 0:  # Log every 30 frames
                    log_detection_with_location(detection_count, confidence_scores)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)

            # Yield the JPEG data to Flask
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            if cv2.waitKey(1) == 27 or terminate_flag:  # Exit when ESC key is pressed or terminate flag is set
                break
        else:
            # Break the loop if the video file capture fails
            print("Failed to read frame, breaking loop")
            break
    cap.release()
    os._exit(0)  # Terminate the script when the video stream ends or terminate flag is set

# Define a route to serve the video stream
@app.route('/video_feed')
def video_feed():
    file_path = request.args.get('file')
    return Response(generate(file_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get location logs
@app.route('/location_logs')
def location_logs():
    return jsonify(detection_logs)

# Route to get current location
@app.route('/current_location')
def current_location():
    # Prefer GPS location if available
    if gps_location:
        return jsonify(gps_location)
    
    # Fallback to IP location
    client_ip = get_client_ip()
    location_data = get_location_from_ip(client_ip)
    return jsonify(location_data) if location_data else jsonify({"error": "Location not found"})

# Route to save GPS location from mobile device
@app.route('/save_gps_location', methods=['POST'])
def save_gps_location():
    global gps_location
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        accuracy = data.get('accuracy')
        timestamp = data.get('timestamp')
        
        if latitude and longitude:
            # Get address from coordinates
            address = get_address_from_coords(latitude, longitude)
            
            gps_location = {
                'source': 'GPS',
                'latitude': latitude,
                'longitude': longitude,
                'accuracy': f"±{accuracy:.0f} meters" if accuracy else "Unknown",
                'address': address,
                'timestamp': timestamp
            }
            
            print(f"GPS location saved: {latitude}, {longitude} (±{accuracy:.0f}m)")
            
            return jsonify({
                'status': 'success',
                'message': 'GPS location saved',
                'address': address
            })
        else:
            return jsonify({'error': 'Invalid coordinates'}), 400
            
    except Exception as e:
        print(f"Error saving GPS location: {e}")
        return jsonify({'error': 'Failed to save location'}), 500

# Route to get GPS status
@app.route('/gps_status')
def gps_status():
    return jsonify({
        'has_gps': gps_location is not None,
        'location': gps_location
    })

# Define a route to serve the HTML page with the file upload form
@app.route('/', methods=['GET', 'POST'])
def index():
    global terminate_flag
    if request.method == 'POST':
        if request.form.get("camera") == "true":
            file_path = "camera"
        elif request.form.get("ngrok") == "true":
            file_path = "ngrok"
        elif 'file' in request.files:
            file = request.files['file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
        else:
            file_path = None
        return render_template('index.html', file_path=file_path)
    else:
        terminate_flag = False
        return render_template('index.html')

# Firebase integration routes
@app.route('/sync_to_firebase', methods=['POST'])
def sync_to_firebase():
    """
    Manually sync all detection logs to Firebase
    """
    if not detection_logs:
        return jsonify({'error': 'No detections to sync'}), 400
    
    result = batch_upload_detections_to_firebase(detection_logs)
    
    if result:
        return jsonify({
            'success': True,
            'message': 'Detections synced to Firebase',
            'details': result
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to sync detections'
        }), 500


@app.route('/firebase_status', methods=['GET'])
def firebase_status():
    """
    Check Firebase connection status
    """
    try:
        response = requests.get(UPLOAD_ENDPOINT, timeout=5)
        
        return jsonify({
            'status': 'connected' if response.status_code in [200, 404] else 'disconnected',
            'dashboard_url': DASHBOARD_URL,
            'endpoint': UPLOAD_ENDPOINT,
            'response_code': response.status_code
        })
    except Exception as e:
        return jsonify({
            'status': 'disconnected',
            'error': str(e),
            'dashboard_url': DASHBOARD_URL
        })


@app.route('/stop', methods=['POST'])
def stop():
    global terminate_flag
    terminate_flag = True
    return "Process has been Terminated"

if __name__ == '__main__':
    app.run(debug=True)
