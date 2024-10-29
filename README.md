# IPL Win Predictor 🏏🔮

Welcome to the IPL Win Predictor! This is an end-to-end machine learning project focused on predicting win probabilities in the Indian Premier League (IPL). From data collection to model building, this project demonstrates the application of machine learning in predicting outcomes in cricket matches.

# 🚀 Key Features

	•	Machine Learning Model: Predicts the win probability of an IPL match based on historical data and match features.
	•	Interactive Web App: Provides match predictions via a simple web interface using Flask.
	•	Pretrained Model: A pre-built pipeline stored in pipe.pkl for efficient predictions.
	•	Dataset: Based on the IPL dataset, which contains historical IPL match data.
    link: https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set

# 📂 Repository Structure

📦ipl_win_predictor
 ┣ 📜README.md             # Project documentation
 ┣ 📜app.py                # Flask web application for prediction
 ┣ 📜ipl_win_predictor.ipynb  # Jupyter notebook for model development
 ┣ 📜pipe.pkl              # Pretrained model pipeline for match prediction
 ┣ 📜requirements.txt      # Python dependencies and libraries
 ┗ 📜.gitignore            # Files and directories to ignore in git

 # 🧠 How It Works

The IPL Win Predictor uses a machine learning model trained on the IPL dataset. The project focuses on:

	1.	Data Processing:
	•	Data on team performance, player stats, toss outcomes, venue effects, and other match details are preprocessed and tokenized.
	2.	Model Training:
	•	The model is trained to predict match outcomes by analyzing key match features and calculating win probabilities.
	3.	Prediction:
	•	The web interface accepts match details, and the model (pipe.pkl) predicts the winning team based on the inputs.

# 🔧 Usage

	1.	Web Interface:
	•	Run the Flask app (app.py) and visit http://127.0.0.1:5000/ to input match details and receive a win probability prediction.
	2.	Jupyter Notebook:
	•	Open ipl_win_predictor.ipynb to explore the model development process, including data preprocessing and training steps.
