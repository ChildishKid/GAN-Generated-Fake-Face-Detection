import bentoml
import numpy as np
import tensorflow as tf
from PIL import Image
from bentoml.artifact import TensorflowSavedModelArtifact
from bentoml.handlers import ImageHandler

@bentoml.env(auto_pip_dependencies=True)
@bentoml.artifacts([TensorflowSavedModelArtifact('model')])
class GeneratedFaceClassifier(bentoml.BentoService):
    
    @bentoml.api(ImageHandler)
    def predict(self, image):
        img = Image.fromarray(image).convert("RGB").resize((1024, 1024))
        img = np.asarray(img).astype('float32')
        img = img / 255.0
        img = (np.expand_dims(img, 0))
        
        infer = self.artifacts.model.signatures['serving_default']
        prediction = infer(tf.convert_to_tensor(img))
        prediction_classified = prediction['output_1'].numpy().argmax(axis=-1)
        
        if prediction_classified == 0:
            return {"Classification": "Fake", "Score": prediction['output_1'].numpy()[0][0]}
        else:
            return {"Classification": "Real", "Score": prediction['output_1'].numpy()[0][1]}
