from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import os
import pandas as pd

app = Flask(__name__)
CORS(app)

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
        
        try:
            # Note: Using DeepFace.find() to find the image in the reference images
            print("before find")
            result_df = DeepFace.find(img_path=file_path, db_path=REFERENCE_IMAGES_FOLDER)
            print(result_df.head())
            print("after find")
            if result_df.shape[0] == 0:
                # No matches found
                return jsonify({"message": "Face not recognized", "results": []})
            else:
                # Matches found, process and return the results
                results = result_df.to_dict('records')  # Converts DataFrame to a list of dicts
                return jsonify({"message": "Face recognized", "results": results})
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
