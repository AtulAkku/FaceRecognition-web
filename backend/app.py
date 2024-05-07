from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2
import os
import numpy as np

app = Flask(__name__)
CORS(app)

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

# Folder where uploaded images will be saved
UPLOAD_FOLDER = 'uploaded_images'
# Ensure the folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Folder containing reference images for comparison
REFERENCE_IMAGES_FOLDER = 'reference_images'

@app.route('/verify', methods=['POST'])
def verify_face():
    file = request.files['file']
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        results = []
        face_recognized = False  # Indicates if any face matched
        
        for person_folder in os.listdir(REFERENCE_IMAGES_FOLDER):
            person_folder_path = os.path.join(REFERENCE_IMAGES_FOLDER, person_folder)
            if os.path.isdir(person_folder_path):
                for ref_image in os.listdir(person_folder_path):
                    ref_image_path = os.path.join(person_folder_path, ref_image)
                    try:
                        result = DeepFace.verify(file_path, ref_image_path, model_name = models[2])
                        print(result)
                        if result['verified'] == True:
                            results.append({"reference_image": ref_image, **result})
                            face_recognized = True
                    except Exception as e:
                        app.logger.error(f"Error processing images: {str(e)}")
                        return jsonify({"error": str(e)}), 500
        
        if not face_recognized:
            # Optionally, adjust your response to indicate non-recognition
            return jsonify({"message": "Face not recognized", "results": results, "verified": False})
        else:
            return jsonify({"message": "Face recognized", "results": results, "verified": True})

@app.route('/analyze', methods=['POST'])
def analyze_face():
    file = request.files['file']
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        try:
            # Perform the analysis
            analysis_result = DeepFace.analyze(img_path=file_path, actions=['age', 'gender', 'emotion'])
            return jsonify(analysis_result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
