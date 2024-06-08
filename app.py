import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import datasets
from PIL import Image

# Load the Iris dataset (for information display)
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])

# Load your model
with open('finalized_model.pickle', 'rb') as file:
    model = pickle.load(file)

# Set the page configuration
st.set_page_config(page_title='Iris Flower Classification',
                   page_icon='🌸', layout='centered', initial_sidebar_state='expanded')

# Add a title and introduction
st.title("🌸 Iris Flower Classification")
st.write("""
Welcome to the Iris Flower Classification app. 
This app uses a machine learning model to classify Iris flowers into three species: 
- Setosa
- Versicolor
- Virginica

You can adjust the feature values using the sliders on the sidebar and see the classification result in real-time.
""")

# Sidebar - User inputs
st.sidebar.header("Input Features")

# Function to get user input features
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.0, 0.1)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.0, 0.1)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 1.5, 0.1)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.5, 0.1)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input
st.subheader('User Input Parameters')
st.write(input_df)

# Predict the class
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display the prediction
st.subheader('Prediction')
st.write(f"The model predicts that the Iris flower is: **{prediction[0]}**")  # Directly use the predicted species

# Display the probability
st.subheader('Prediction Probability')
proba_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.write(proba_df)

# Display some additional info or visualizations
st.subheader('Iris Dataset Overview')
st.write('Below is a preview of the Iris dataset used to train the model.')
st.dataframe(iris_df.head(10))

# Footer
st.markdown("""
---
**Created by [Pranjal Thakre]**
""")





