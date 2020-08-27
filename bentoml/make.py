import tensorflow as tf
from generated_face_classifier import GeneratedFaceClassifier

model = tf.saved_model.load('../saved_models/ResNet50_base/')
classifier = GeneratedFaceClassifier().pack('model', model)
saved_path = classifier.save()