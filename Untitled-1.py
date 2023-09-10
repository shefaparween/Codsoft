# Example using OpenCV and Haar Cascade Classifier
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Example using a Siamese network for face recognition
from siamese_model import SiameseNetwork

siamese_net = SiameseNetwork()
siamese_net.load_weights('siamese_model_weights.h5')

# Generate embeddings for detected faces
face_embeddings = []
for face in detected_faces:
    embedding = siamese_net.generate_embedding(face)
    face_embeddings.append(embedding)

# Compare embeddings for recognition
for i, embedding in enumerate(face_embeddings):
    similarity = compare_embeddings(embedding, known_embeddings)
    if similarity > threshold:
        recognized_name = known_names[i]
        print(f"Recognized: {recognized_name}")
# Example using a Siamese network for face recognition
from siamese_model import SiameseNetwork

siamese_net = SiameseNetwork()
siamese_net.load_weights('siamese_model_weights.h5')

# Generate embeddings for detected faces
face_embeddings = []
for face in detected_faces:
    embedding = siamese_net.generate_embedding(face)
    face_embeddings.append(embedding)

# Compare embeddings for recognition
for i, embedding in enumerate(face_embeddings):
    similarity = compare_embeddings(embedding, known_embeddings)
    if similarity > threshold:
        recognized_name = known_names[i]
        print(f"Recognized: {recognized_name}")
