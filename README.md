# Indian License Plate Detection System

A Flask-based web application for detecting and recognizing Indian license plates using computer vision and OCR techniques.

![Project Logo](https://img.icons8.com/color/96/000000/car.png)

## Overview

This project implements a license plate detection system specifically designed for Indian license plates. It uses a combination of computer vision techniques and OCR (Optical Character Recognition) to detect and read license plate numbers from images or live camera feed.

## Features

- 🚗 Real-time license plate detection
- 📸 Image upload and processing
- 🔍 OCR text recognition
- 📊 State code detection
- 💾 SQLite database for storing detection history
- 🌐 Web-based interface
- 📱 Responsive design

## Project Structure

```
Number_plate/
├── app.py                 # Flask application
├── database.py            # SQLite database operations
├── plate_detector.py      # License plate detection logic
├── state_codes.py         # Indian state codes mapping
├── requirements.txt       # Python dependencies
├── uploads/              # Directory for uploaded images
├── templates/            # HTML templates
│   ├── index.html        # Detection page
│   ├── landing.html      # Landing page
│   └── history.html      # Detection history page
└── license_plates.db     # SQLite database
```

## Setup Instructions

1. Clone the repository:

```bash
git clone git@github.com:SudarshanProCoder/Indian-License-Plate-Detection-System.git
cd Indian-License-Plate-Detection-System
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python app.py
```

5. Open your browser and navigate to:

```
http://localhost:5000
```

## Usage

1. **Home Page**

   - Navigate to the home page to learn about the project
   - View features and capabilities

2. **Detection Page**

   - Upload an image or use live camera feed
   - View detected license plate and state
   - See processed image with detection overlay

3. **History Page**
   - View all previous detections
   - Filter by state
   - See detection timestamps and images

## Output Examples

### Landing Pages

![Landing Page](images/img2.png)
![Landing Page](images/img3.png)

### 1. Detection Interface

![Detection Interface](images/img4.png)
_The main detection interface with image upload and live camera options_

### 2. Detection Result

![Detection Result](images/img1.png)
_Example of a detected license plate with state information_

### 3. History View

![History View](images/img5.png)
_The history page showing previous detections_

## Technical Details

- **Backend**: Flask, OpenCV, EasyOCR
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Database**: SQLite
- **OCR Engine**: EasyOCR with custom preprocessing
- **Detection Algorithm**: Custom plate detection with contour analysis

## Developer Information

- **Developer**: Sudarshan date
- **Development Date**: March 2025
- **Contact**: sudarshandate21@gmail.com

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- EasyOCR for text recognition
- Flask for web framework
- Bootstrap for frontend components

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For support, please open an issue in the repository or contact the developer.
