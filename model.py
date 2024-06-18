<<<<<<< HEAD
print("hi")
print('hello')
=======
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image


# Function to load and preprocess the image.
def load_image(image_path):
    
    # Load the image.
    image = Image.open(image_path)
    
    # Resize the image.
    img = image.resize((224, 224))
 
    # Convert image to numpy array.
    img_array = np.array(img)

    # Normalize the image array.
    img_array = img_array / 255.0
    
    # Add a batch dimension.
    image_tensor = np.expand_dims(img_array, axis=0)
    
    return image_tensor





def predict_image(model, image_path):
    
    
    # Load and preprocess the image
    image_tensor = load_image(image_path)
    
    # Predict the class probabilities
    probability_prediction = model.predict( image_tensor)

    return np.round(probability_prediction, decimals=4)

>>>>>>> 3cd38bd813b3228d354c1edebb42de73482b2b4f
