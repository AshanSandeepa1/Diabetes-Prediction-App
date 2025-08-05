# Diabetes Prediction Web App

This is a user-friendly machine learning web application that predicts the likelihood of diabetes based on key medical parameters. Built with Streamlit, the app provides a simple interface for both data exploration and real-time prediction using trained ML models.

## Live App

Visit the deployed app here:  
[https://diabetes-prediction-app-ashan.streamlit.app](https://diabetes-prediction-app-ashan.streamlit.app)

## Features

- Clean and interactive homepage with detailed guidance
- View and explore the training dataset
- Visualize trends and correlations using dynamic charts
- Input health parameters to get instant diabetes predictions
- Display of model confidence score
- Performance comparison of multiple ML models
- Responsive design with light/dark theme support
- Error handling, input validation, and loading animations included

## Tech Stack

- **Frontend & App**: [Streamlit](https://streamlit.io)
- **Backend/Modeling**: Python, scikit-learn, pandas, matplotlib, seaborn
- **Deployment**: Streamlit Cloud
- **Version Control**: Git + GitHub

## How to Run Locally

# Method 1
1. Clone the Repository
   git clone https://github.com/AshanSandeepa1/Diabetes-Prediction-App.git
   cd your-repo-name

2. Create a Virtual Environment (Optional but Recommended)
    python -m venv venv
    source venv/bin/activate     # On Windows: venv\Scripts\activate

3. Install Dependencies
    pip install -r requirements.txt

4. Run the App
   streamlit run app.py

# Method 2 - Using Docker
1. Clone the Repository
   git clone https://github.com/AshanSandeepa1/Diabetes-Prediction-App.git
   cd your-repo-name

2. Open the project root directory in CLI.
   
4. Make sure Docker is installed in your machine.
   
6. Run the App
   docker-compose up --build


## Project Structure

your-project/
├── app.py 
├── requirements.txt
├── model.pkl
├── assets/
├── data/
│   └── dataset.csv
├── pages/
│   ├── Home.py
│   ├── 1_Explore_Data.py
│   ├── 2_Data_Visualization.py
│   ├── 3_Predict_Diabetes.py
│   └── 4_Model_Performance.py
├── notebooks/
│   └── model_training.ipynb
└── README.md


## License
This project is open-source and available under the MIT License.


