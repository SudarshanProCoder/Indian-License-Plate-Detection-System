import cv2
import numpy as np
import easyocr
import re
from ultralytics import YOLO

class LicensePlateDetector:
    def __init__(self):
        # Initialize YOLO model with pre-trained license plate detection weights
        self.model = YOLO('license_plate_detector.pt')
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'])
        
        # Initialize video capture
        self.cap = None
        
        # Detection confidence threshold
        self.conf_threshold = 0.3  # Lowered threshold for better detection
        
        # Indian license plate pattern (e.g., RJ14CV0002)
        self.plate_pattern = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$')

    def preprocess_for_ocr(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise while keeping edges sharp
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Apply dilation to connect text components
        dilated = cv2.dilate(opening, kernel, iterations=1)
        
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(dilated)
        
        # Apply additional sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Apply additional preprocessing for better OCR
        # Resize the image to a standard size
        resized = cv2.resize(sharpened, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Apply additional thresholding
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply additional noise removal
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        return denoised

    def clean_plate_text(self, text):
        # Remove all non-alphanumeric characters
        cleaned = ''.join(c for c in text if c.isalnum())
        
        # Remove any text that's too long (likely not a license plate)
        if len(cleaned) > 15:
            return None
        
        # Try to fix common OCR mistakes
        replacements = {
            '0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A',
            '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G',
            'O': '0', 'I': '1', 'Z': '2', 'B': '8', 'A': '4',
            'S': '5', 'G': '6', 'T': '7'
        }
        
        # Try different combinations of replacements
        possible_texts = [cleaned]
        for i in range(len(cleaned)):
            if cleaned[i] in replacements:
                new_text = cleaned[:i] + replacements[cleaned[i]] + cleaned[i+1:]
                possible_texts.append(new_text)
        
        # Check each possible text against the pattern
        for text in possible_texts:
            if self.plate_pattern.match(text):
                return text
        
        return None

    def detect_plate(self, image_path=None, frame=None):
        if image_path:
            # Read the image
            img = cv2.imread(image_path)
        else:
            img = frame
            
        if img is None:
            return None, None, "Failed to read image"
        
        # Resize image for better detection
        height, width = img.shape[:2]
        if width > 1280:
            scale = 1280 / width
            img = cv2.resize(img, (1280, int(height * scale)))
        
        # Perform detection with YOLO
        results = self.model(img, conf=self.conf_threshold)
        
        if len(results) > 0:
            result = results[0]
            if len(result.boxes) > 0:
                # Get all detected objects
                boxes = result.boxes
                
                # Find the license plate box (class 0 in the license plate model)
                plate_boxes = [box for box in boxes if box.cls == 0]
                
                if plate_boxes:
                    # Get the plate box with highest confidence
                    plate_box = plate_boxes[0]
                    x1, y1, x2, y2 = map(int, plate_box.xyxy[0])
                    
                    # Extract the plate region with some padding
                    padding = 20  # Increased padding
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(img.shape[1], x2 + padding)
                    y2 = min(img.shape[0], y2 + padding)
                    plate = img[y1:y2, x1:x2]
                    
                    # Preprocess the plate for OCR
                    processed_plate = self.preprocess_for_ocr(plate)
                    
                    # Perform OCR with EasyOCR
                    try:
                        # Read text from the processed plate
                        results = self.reader.readtext(processed_plate)
                        
                        # Get the text with highest confidence
                        best_text = None
                        best_confidence = 0
                        
                        for (bbox, text, prob) in results:
                            if prob > best_confidence:
                                cleaned_text = self.clean_plate_text(text)
                                if cleaned_text:
                                    best_text = cleaned_text
                                    best_confidence = prob
                        
                        if best_text:
                            # Draw bounding box and text on the original image
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, best_text, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            return best_text, img, None
                        else:
                            return None, img, "Could not read text from number plate"
                    except Exception as e:
                        return None, img, f"OCR Error: {str(e)}"
        
        return None, img, "No license plate detected"

    def start_camera(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return self.cap.isOpened()

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Detect license plate in the frame
        text, frame_with_detection, error = self.detect_plate(frame=frame)
        return frame_with_detection, text, error

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

# Example usage
if __name__ == "__main__":
    detector = LicensePlateDetector()
    text, plate_img, error = detector.detect_plate("test_image.jpg")
    if text:
        print(f"Detected license plate: {text}")
    else:
        print(f"Error: {error}") 