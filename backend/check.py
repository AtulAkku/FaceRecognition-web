from deepface import DeepFace

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

UPLOAD_FOLDER = 'uploaded_images/user-image.jpg'
# Folder containing reference images for comparison
# REFERENCE_IMAGES_FOLDER = 'reference_images'

result_df = DeepFace.verify(UPLOAD_FOLDER,'reference_images/GIINT240005/IMG_20240318_120948.jpg', model_name = models[2])

print(result_df)