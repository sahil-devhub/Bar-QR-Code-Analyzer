# Import necessary libraries
import cv2
import os
import base64
import requests
import json
import numpy as np
import time
import threading
from pyzbar.pyzbar import decode as zbar_decode

#---
# Helper function to convert the image frame to base64 for API call
def frame_to_base64(frame):
    """
    Converts a numpy array image frame to a base64 encoded string.
    
    Args:
        frame (np.array): The input image frame.
    
    Returns:
        str: The base64 encoded string of the image.
    """
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# ---
# Function to get the camera index
def get_camera_index():
    """
    Finds and returns the index of the first available webcam.
    """
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return i
    print("Error: Could not find any camera. Please check your setup.")
    return -1

# ---
# Function to decode barcodes and QR codes from a saved image using Gemini API
def decode_with_gemini(frame, api_key):
    """
    Sends a frame to the Gemini API to decode a barcode or QR code.
    
    Args:
        frame (np.array): The image frame to decode.
        api_key (str): Your Google AI Studio API key.
    
    Returns:
        str or None: The decoded ID if successful, otherwise None.
    """
    try:
        encoded_string = frame_to_base64(frame)

        headers = {
            "Content-Type": "application/json",
        }

        prompt = "You are a highly specialized and accurate barcode and QR code scanner. Your sole purpose is to read the alphanumeric ID contained within the code in the image. You must return only the raw ID, with absolutely no other text, explanations, punctuation, or formatting. If the image contains a code, provide its ID. If no code is found, respond with nothing."

        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        },
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": encoded_string
                            }
                        }
                    ]
                }
            ]
        }
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
        response.raise_for_status()
        
        result = response.json()

        if 'candidates' in result and len(result['candidates']) > 0:
            text_part = result['candidates'][0]['content']['parts'][0]['text']
            clean_id = text_part.strip().replace("`", "").replace("json", "").replace("\n", "")
            
            if clean_id:
                print(f"Successfully fetched ID: {clean_id}")
    
    except requests.exceptions.Timeout:
        print("Error: The API request timed out.")
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
    except Exception as e:
        print(f"An error occurred with Gemini API: {e}")

# ---
# Main function for the hybrid scanner
def hybrid_scanner(api_key):
    """
    Starts a webcam feed and continuously scans with Pyzbar.
    If Pyzbar fails, a key press ('s') can trigger a Gemini API fallback.
    
    Args:
        api_key (str): Your Google AI Studio API key.
    """
    print("Initializing camera...")
    start_time = time.time()
    camera_index = get_camera_index()
    if camera_index == -1:
        return

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Could not open video stream for camera index {camera_index}.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5) 
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    time.sleep(1)

    for _ in range(5):
        cap.read()
    
    print(f"Camera started in {time.time() - start_time:.2f} seconds.")
    print("Scanning for codes in real-time. Press 's' for Gemini fallback or 'q' to quit.")

    decoded_ids = set()
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from the camera.")
            break
        
        last_frame = frame.copy()
            
        decoded_objects = zbar_decode(frame)
        
        if decoded_objects:
            for obj in decoded_objects:
                code_data = obj.data.decode('utf-8')
                
                if code_data not in decoded_ids:
                    print(f"Successfully fetched ID: {code_data}")
                    decoded_ids.add(code_data)
                    
                points = obj.polygon
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([p for p in points], dtype=np.float32))
                    hull = np.int32(hull)
                    cv2.polylines(frame, [hull], True, (0, 255, 0), 2)
                else:
                    cv2.polylines(frame, [np.int32(points)], True, (0, 255, 0), 2)
                        
                cv2.putText(frame, code_data, (obj.rect.left, obj.rect.top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(frame, "Scanning... Press 's' for photo or 'q' to quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Hybrid Scanner", frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            break
            
        elif key == ord('q'):
            print("Quitting...")
            last_frame = None
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if last_frame is not None and key == ord('s'):
        print("Fetching ID...")
        thread = threading.Thread(target=decode_with_gemini, args=(last_frame, api_key))
        thread.start()

    return None

if __name__ == "__main__":
    API_KEY = "YOUR_API_KEY"
    
    if API_KEY == "YOUR_API_KEY":
        print("Please replace 'YOUR_API_KEY' with your actual Gemini API key.")
    else:
        hybrid_scanner(API_KEY)

