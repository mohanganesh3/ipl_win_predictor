import streamlit as st
import pickle
import pandas as pd

# Team and city lists
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Daredevils'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load the trained model
with open('pipe.pkl', 'rb') as file:
    pipe = pickle.load(file)
st.title('IPL Win Predictor')

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

# Target input
target = st.number_input('Target', min_value=1, step=1)

# Columns for score, overs, and wickets
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=9, step=1)

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets

    # Ensure that inputs are within valid ranges
    if runs_left < 0 or balls_left < 0:
        st.error("Invalid input: Score exceeds target or overs exceed the match limit.")
    else:
        crr = score / overs if overs > 0 else 0  # Current run rate
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0  # Required run rate

        # Create a DataFrame for model input
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Predict the probabilities
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Display the results
        st.header(f"{batting_team} - {round(win * 100, 2)}% chance of winning")
        st.header(f"{bowling_team} - {round(loss * 100, 2)}% chance of winning")