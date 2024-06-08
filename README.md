
# Iris Flower Classification using Machine Learning

![Iris Flower](images/iris_flower.jpg)

## Overview

This project implements a machine learning model to classify Iris flowers into three species: Setosa, Versicolor, and Virginica. The classification is based on the flower's sepal and petal dimensions. The model is deployed as an interactive web application using Streamlit.

## Key Features

- **Interactive Input**: Users can adjust sliders to input sepal and petal dimensions.
- **Real-Time Predictions**: The app instantly predicts the species and provides the probability for each class.
- **Visual Insights**: Each prediction is accompanied by an image of the predicted Iris species and an overview of the dataset.
- **User-Friendly Interface**: Built with a clean and intuitive layout, making it accessible for everyone.

## Technologies Used

- **Python**: For data processing and machine learning.
- **Scikit-Learn**: For training the classification model.
- **Streamlit**: To create a dynamic and interactive web app.
- **Pandas & NumPy**: For data handling and manipulation.

## Getting Started

### Prerequisites

- Python 3.x installed
- Python libraries: scikit-learn, streamlit, pandas, numpy

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/iris-classification-streamlit.git
    ```

2. Navigate to the project directory:

    ```bash
    cd iris-classification-streamlit
    ```

3. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to http://localhost:8501 to access the app.

### Dataset

The Iris dataset used for training the model is available in the `data` directory. The dataset contains four features: sepal length, sepal width, petal length, and petal width.

### Directory Structure

```
iris-classification-streamlit/
│
├── app.py                # Streamlit web application
├── data/                 # Dataset directory
│   └── iris.csv          # Iris dataset (CSV format)
├── images/               # Images directory
│   └── iris_flower.jpg   # Image for README
├── Finalized_model.pickle  # Pickle file containing the trained model
├── README.md             # Project README file
└── requirements.txt      # Python dependencies
```

## Acknowledgements

- This project was completed as part of the Bharat Intern program.
- The Iris dataset used in this project is a well-known dataset in the machine learning community.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
