import cv2
import os
import numpy as np
from PIL import Image

# Initialize the recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Check if the detector was loaded correctly
if detector.empty():
    raise Exception("Error loading cascade classifier.")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
    
    if not imagePaths:
        print("Error: No image files found in 'TrainingImage'.")
        exit()

    print("Files in directory:", os.listdir(path))  # Print files in the directory for debugging

    faceSamples = []
    Ids = []

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')  # Convert image to grayscale
        imageNp = np.array(pilImage, 'uint8')

        # Extract ID from the filename (assuming filenames are like subject.1.jpg)
        try:
            Id = int(os.path.split(imagePath)[-1].split(".")[0])  # Assuming filenames are like subject.1.jpg
        except ValueError:
            print(f"Skipping invalid image file: {imagePath}")
            continue

        # Detect faces in the image
        faces = detector.detectMultiScale(imageNp)
        if len(faces) == 0:
            print(f"No faces detected in image {imagePath}. Skipping.")
            continue  # Skip images with no detected faces

        # Debugging: Show the detected faces in the image
        print(f"Faces detected in {imagePath}: {faces}")
        for (x, y, w, h) in faces:
            cv2.rectangle(imageNp, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle around the face
        cv2.imshow("Detected Faces", imageNp)
        cv2.waitKey(0)  # Wait for a key press to close the window

        # Append faces and corresponding Id to the lists
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)

    cv2.destroyAllWindows()  # Close any open OpenCV windows
    print(f"Collected {len(faceSamples)} face samples for {len(set(Ids))} unique subjects.")
    
    if len(faceSamples) == 0:
        print("Error: No face samples collected.")
        exit()

    return faceSamples, Ids

# Collect faces and IDs from the 'TrainingImage' directory
faces, Ids = getImagesAndLabels('TrainingImage')

# Check if we have enough training data
if len(faces) < 2:
    print("Error: Not enough training data. Please add more images.")
    exit()

# Train the recognizer
recognizer.train(faces, np.array(Ids))

# Save the trained model
recognizer.save('trainer.yml')

print("Training completed and model saved.")
