import os 
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from model import predict_image
from tensorflow.keras.models import load_model
from auth import hash_password, get_db_connection, init_db, add_user, authenticate_user, add_patient, add_dr_prediction, get_patient_data, fetch_predictions, get_patient_id, generate_pdf_report



# ==================== Set the page configuration ====================#

st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon=":eye:",
    layout="wide",
    # initial_sidebar_state="expanded",
)



# ==================== Styling ==================== #


st.markdown(
    """
    <style>
        .title-text {
            background: linear-gradient(to right, #ff69b4, #9400d3); /* Vibrant pink/purple gradient */
            color: white; /* White text for the header */
            padding: 25px; /* Increased padding for more space */
            text-align: center;
            border-radius: 10px; /* Rounded corners for a softer look */
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Subtle shadow for depth */
        }
        
        .title-text h1{
            font-size: 3rem; /* Larger font size for the header */
        }
        
        .footer{
            background-color: #A52A2A;
            color: #FFFFFF;
            padding: 5px;
            position; fixed;
            text-align: center;
            bottom: 0;
            width: 100%;
            height: 50px;
        }
        
    </style>
    """,
    unsafe_allow_html=True
    )



# ==================== Header ==================== #

st.markdown("""
            <div class="title-text">
            <h1>Automated Detection of Diabetic Retinopathy</h1>
            </div>
            """, 
            unsafe_allow_html=True
            )




# ==================== Spacing ==================== #
st.markdown("&nbsp;", unsafe_allow_html=True)




# ==================== Styling ==================== #

st.markdown(
    """
    <style>
        .about-text, .get-started-section {
            # background-color: #fff; /* White background */
            border: 1px solid #ddd; /* Subtle border */
            padding: 25px; /* More padding for better readability */
            border-radius: 10px; /* Rounded corners for a softer look */
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Subtle shadow for depth */
        }

        .about-text h2, .get-started-section h3 {
            color: #fffff;
            margin-bottom: 15px;
        }

        .about-text p, .get-started-section p {
            line-height: 1.5;
        }
    </style>
    """,
    unsafe_allow_html=True,
)




# ==================== About the App Section =================== #


with st.container():
    
    with st.expander("About the App"):
        
        st.markdown(
            """
            <div class="about-text">
                <p>
                This application automates the early-stage detection of diabetic retinopathy (DR) using machine learning and computer vision. It analyzes retinal fundus images to provide 
                accurate, consistent, and efficient identification of DR, supporting early intervention and personalized treatment. By leveraging these techniques, it addresses challenges
                posed by subjective assessments and manual labor in existing methods. The convolutional neural networks (CNN) model achieves an impressive accuracy and a 96% recall score, 
                ensuring effective detection of diabetic retinopathy cases amidst rising diabetes prevalence and ophthalmologist shortages.
                </p>            
            </div>
            """,
            unsafe_allow_html=True
            )
        
        
        

# ==================== Get Started Section =================== #

with st.container():
    st.markdown(
        """
        <div class="get-started-section">
            <h3>Get Started</h3>
            <p>
            Just upload your retinal fundus image, and the system will quickly analyze it to detect the presence or absence of diabetic retinopathy (DR) along with the percentage level of 
            confidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True
        )




# ==================== Spacing ==================== #

st.markdown("&nbsp;")
st.markdown("&nbsp;")




# ==================== Create Tabs ==================== #

# Define tabs with emojis
tab1, tab2 = st.columns(2)

# Objectives tab
with tab1:
    st.write("🎯 Objectives")
    with st.expander("Click to see objectives"):
        st.markdown(
            """
            <ul>
                <li>Automate the detection of Diabetic Retinopathy (DR) using machine learning and computer vision.</li>
                <li>Support early intervention by providing timely and reliable screening results.</li>
                <li>Reduce reliance on subjective assessments and manual labor for DR diagnosis.</li>
                <li>Address challenges posed by the increasing prevalence of diabetes and the shortage of ophthalmologists.</li>
                <li>Develop a robust and scalable system for DR detection.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

# Features tab
with tab2:
    st.write("📱 App Features")
    with st.expander("Click to see features"):
        st.markdown(
            """
            <ul>
                <li>Upload a single or batches of retinal fundus images for DR analysis.</li>
                <li>Utilize machine learning and computer vision techniques for analysis.</li>
                <li>Display accurate detection results.</li>
                <li>Provide confidence levels in the presence or absence of DR detection.</li>
                <li>Capability for signing in and signing up.</li>
                <li>Storing and retrieving patient information.</li>
                <li>Store prediction information for tracking and download prediction reports in PDF format.</li>
                <li>Ensure user privacy and security of patient's data.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )




# ==================== Spacing ==================== #
st.markdown("&nbsp;")
st.markdown("&nbsp;")


st.write("Please use the sidebar to Sign in or Sign up in order to use the app!")

# Initialize the database
init_db()


# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# Sidebar for user authentication
st.sidebar.title("User Authentication")


# Tabs for login and signup
tab1, tab2 = st.sidebar.tabs(["Sign in", "Sign Up"])

with tab1:
    st.subheader("Sign in")
    login_username = st.text_input("Username", key="login_username")
    login_password = st.text_input("Password", type="password", key="login_password")
    login_button = st.button("Sign in")
    
    if login_button:
        user = authenticate_user(login_username, login_password)
        if user:
            st.success(f"Welcome, {user[2]}!")
            st.session_state.logged_in = True
            st.session_state.username = user[1]
        else:
            st.error("Invalid username or password!")

with tab2:
    st.subheader("Sign Up")
    signup_name = st.text_input("Full Name", key="signup_name")
    signup_username = st.text_input("Username", key="signup_signup_username")
    signup_password = st.text_input("Password", type="password", key="signup_signup_password")
    signup_email = st.text_input("Email", key="signup_email")
    signup_button = st.button("Sign Up")
    
    if signup_button:
        add_user(signup_username, signup_name, signup_password, signup_email)
        st.success("User registered successfully! Please Sign in.")




# Show a message if the user is logged in
if st.session_state.logged_in:
    st.write(f"Signed in as: {st.session_state.username}")
    st.markdown("&nbsp;")
      
        
    # ==================== Image Input Section ==================== #


    # File Uploader
    uploaded_images = st.file_uploader("Upload retinal fundus images for analysis", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Load the model.
    model = load_model("model-folder/diabetic-retino-model.h5")

    # Display Uploaded Image and Classification Results
    if uploaded_images is not None:
        
        # Predict Diabetic Retinopathy.
        button_key = "identify_diagnosis_button"  # Unique key for the button
                    
        if st.button(label='Identify Diagnosis', key=button_key):
        
            # Loop through the uploaded images.
            for uploaded_image in uploaded_images:
                
                if uploaded_image.type in ["image/jpg", "image/jpeg", "image/png"]:
            
                    # Display loading spinner
                    with st.spinner('Classifying...'):
                        
                        image = Image.open(uploaded_image)
                        st.image(image, caption="Uploaded Image")
                    
                        # Save the uploaded image to the working directory.
                        image_path = os.path.join("diabetic_retinopathy_dataset", "uploaded_image.jpg")
                        image.save(image_path)

                        score = []
                    
                        # Perform prediction on the patient's image.
                        confidence_level = predict_image(model=model, image_path=image_path)
                        
                        score.append(confidence_level[0, 0])
                        confidence_score = score[0] * 100
                        
                        # Define the binary class.
                        binary_class = ["DR", "NO-DR"]

                        
                        if confidence_level >= 0.5:
            
                            # Display predicted class and confidence score.
                            st.success(f"The uploaded fundus image shows no signs of Diabetic Retinopathy. My confidence in this prediction is {confidence_score:.2f}%")
                            
                            # Display a bar chart with confidence scores.
                            confidence_scores = [round((100 - confidence_score), 4), round(confidence_score, 4)]
                            st.write("Percentage Level of Confidence")
                            st.bar_chart(pd.DataFrame({
                                "Confidence Score": confidence_scores
                                }, index=binary_class))
                        
                        else: 
                            
                            confidence_score = 100 - confidence_score
                            # Display predicted class and confidence score.
                            st.success(f"The uploaded fundus image indicates the presence of Diabetic Retinopathy. My confidence in this prediction is {confidence_score:.2f}%")
                            
                            # Display a bar chart with confidence scores.
                            confidence_scores = [round(confidence_score, 4), round((100 - confidence_score), 4)]
                            st.write("Percentage Level of Confidence")
                            st.bar_chart(pd.DataFrame({
                                "Confidence Score": confidence_scores
                                }, index=binary_class))
                            
                            
                        # Delete the model path after making prediction.
                        os.remove(image_path)
                
                    
                else:
                    st.error("Please upload a JPG or JPEG or PNG image.", icon="🔴")

    
    st.markdown("---")

    
    patient_data = get_patient_data(login_username)
    
    # Tabs for different sections.
    patient_tab, prediction_tab = st.columns(2)
    
    with patient_tab:
        
        st.subheader("Patient's Information")
        if not patient_data:
            with st.expander("Click to fill patient's info"):
                patient_name = st.text_input("Patient Name")
                patient_age = st.number_input("Patient Age", min_value=0)
                patient_gender = st.selectbox("Patient Gender", ["Male", "Female"])
                patient_contact_info = st.text_input("Contact Info")
                add_patient_button = st.button("Add Patient")
                
                if add_patient_button:
                    add_patient(login_username, patient_name, patient_age, patient_gender, patient_contact_info)
                    st.success("Patient added successfully!")
                    patient_data = get_patient_data(login_username)  # Refresh patient data
        
        if patient_data:
            # Display existing patient information
            st.write(f"Name: {patient_data['name']}")
            st.write(f"Age: {patient_data['age']}")
            st.write(f"Gender: {patient_data['gender']}")
            st.write(f"Contact: {patient_data['contact']}")
        else:
            st.write("No patient data found.")
                    
                    
    with prediction_tab:
        
        st.subheader("Add Prediction Information")
        with st.expander("Click to fill the prediction info for future usage!"):
            
            patient_id = get_patient_id(login_username)
            prediction_class = st.selectbox("Prediction Class", ["DR", "No DR"])
            confidence_score = st.number_input("Confidence Score", min_value=1, max_value=100)
            add_prediction_button = st.button("Add Prediction")
            
            if add_prediction_button:
                add_dr_prediction(patient_id, prediction_class, confidence_score)
                st.success("Prediction added successfully!")
                
                
            # Filter predictions by username
            username_filter = login_username
            if username_filter:
                predictions = fetch_predictions(username_filter)
            
            else:
                predictions = []

            
            # Display predictions
            if predictions:
                st.subheader("Predictions Overview")
                for prediction in predictions:
                    st.write(f"Name: {prediction[0]}")
                    st.write(f"Patient ID: {prediction[1]}")
                    st.write(f"Prediction Class: {prediction[2]}")
                    st.write(f"Confidence Score: {prediction[3]}")
                    st.write(f"Prediction Date: {prediction[4]}")
                    
                st.write("")
                st.write("To get your full predictions overview, click to download the PDF report.")
                    
                    
                # Generate and offer download link for PDF report
                st.markdown("---")
                st.subheader("Download Predictions Report")
                pdf_filename = generate_pdf_report(predictions)
                st.download_button(label="Download PDF Report", data=open(pdf_filename, "rb").read(), file_name=pdf_filename)

        


st.markdown("&nbsp;", unsafe_allow_html=True)
st.markdown("&nbsp;", unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)  # Adding a single line break


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
            padding: 5px;
            margin-top: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            height: 70px;
        
            
        }
    </style>
    """,
    unsafe_allow_html=True
)




# feedback = st.sidebar.text_area("Please share your feedback here:", max_chars=500)

# if st.sidebar.button("Submit"):
#     # Process the feedback (you can save it to a database or send it via email)
#     st.sidebar.success("Thank you for your feedback! It has been submitted successfully.")


