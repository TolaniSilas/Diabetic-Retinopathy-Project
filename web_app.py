import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from model import predict_image
from tensorflow.keras.models import load_model



# Title and Description
st.markdown(
    """
    <style>
        .title-text {
            font-size: 50px;
            font-weight: bold;
            color: #A52A2A;
            margin-bottom: 20px;
        }
        .description-text {
            font-size: 16px;
            color: #FFE4B5;
            margin-bottom: 30px;
            padding: 15px;
            background-color: #333333
        }
        .stApp {
            background-color: #CCCCCC; 
            padding: 20px;
            margin: 15px
        }
        .footer{
            background-color: #A52A2A;
            color: #FFFFFF;
            padding: 5px;
            position; fixed;
            text-align: center;
            bottom: 0;
            width: 100%
        }
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown("<h1 class='title-text'>Animal Classifier Model</h1>", unsafe_allow_html=True)
st.markdown(
    """<p class='description-text'>This model is trained on images featuring four different types of animals: Elephant, Lion, Giraffe, and Zebra. 
    It utilizes Convolutional Neural Networks (CNN) to perform classification, achieving high accuracy and robust performance metrics. This 
    project showcases machines' remarkable capability to discern intricate patterns imperceptible to humans. Not only does this image classification 
    project underscore machines' capacity to analyze images, but it also serves as a demonstration of machine learning's capabilities in image 
    classification. It provides a practical example of how machine learning can address real-world problems, fostering innovation and efficiency 
    across diverse fields.</p>""",
    unsafe_allow_html=True
)


# File Uploader
uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg"])

# Display Uploaded Image and Classification Results
if uploaded_image is not None:
    if uploaded_image.type in ["image/jpg", "image/jpeg"]:
        # Display loading spinner
        with st.spinner('Classifying...'):
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image")
            
            # Save the uploaded image to the working directory.
            image_path = os.path.join("animal_image_classifier_project", "uploaded_image.jpg")
            image.save(image_path)
            
            # Get the trained model file.
            model_file = "animal_image_classifier_project/animal_classifier.pt"

            classes = ["elephant", "giraffe", "lion", "zebra"]

            # Load the model.
            classifier_model = AnimalNet(num_classes=len(classes))
            classifier_model.load_state_dict(torch.load(model_file))
            
            
        
            # Perform prediction
            label = predict_image(model=classifier_model, image_path=image_path)
            
            # Display predicted class and confidence score
            st.write("Predicted Class:", classes[label].title())
            # st.write("Confidence Score:", label["confidence"])
            
            
            # Delete the model path after making prediction.
            os.remove(image_path)
            
            
    else:
        st.error("Please upload a JPG or JPEG image.", icon="🔴")





# ----------------------------------Footer----------------------------------------#


st.markdown(
    """
    <div class="footer"> 
    <p>Copyright 2024 &copy;</p>
    </div>
    """, 
    unsafe_allow_html=True
)








#-------------------------------Add styling to sidebar using CSS styling.------------------------------------#
st.markdown(
    """
    <style>
        .sidebar .sidebar-content{
            background-color: #FFFFFF;
        }
        .h1{
            color: #A52A2A;
            font-size: 16px;
        } 
        .header{
            color: #A52A2A
        } 
        .info {
            padding: 20px;
            border-radius: 10px;
            font-size: 12px;
            background-color: #D3D3D3;
            color: #000000;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .sidebar-close {
            color: #666666; /* Dark Gray */
        }
        .feedback {
            background-color: #D3D3D3;
            font-size: 12px;
            color: #A52A2A;
            padding: 20px;
            margin-top: 20px
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Define the sidebar content using Streamlit's layout components.
st.sidebar.markdown(
    """
    <div class="info">
        <h1 class="h1">About</h1>
        <p>
        This web application is an Animal Classifier project, which aims to classify images of four different types of animals: Elephant, Giraffe,
        Lion, and Zebra. The project utilizes Convolutional Neural Networks (CNNs) to achieve high accuracy in classification. 
        </p>
        <h2 class="header">Purpose</h2>
        <p>
            The purpose of this project is to demonstrate the capabilities of 
            machine learning in image classification and to provide a practical 
            example of how machine learning can be applied to real-world problems.
        </p>
        <h2 class="header">Contributor</h2>
        <p>
            Osunba Silas Tolani
        </p>
        <h2 class="header">Acknowledgments</h2>
        <p>
            I'd like to thank the Streamlit community for their support 
            and the developers of the underlying libraries used in this project.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)



st.sidebar.markdown("""
    <div class="feedback">
        <h1 class="header">Feedback</h1>
    </div>
""", unsafe_allow_html=True)

feedback = st.sidebar.text_area("Please share your feedback here:", max_chars=500)

if st.sidebar.button("Submit"):
    # Process the feedback (you can save it to a database or send it via email)
    st.sidebar.success("Thank you for your feedback! It has been submitted successfully.")


