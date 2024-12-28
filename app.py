from idlelib.configdialog import font_sample_text

import streamlit as st
import pickle
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Add background image with blur effect
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://wallpaperaccess.com/full/1088620.jpg");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

html, body {
    background-color: rgba(0, 0, 0, 0.5);  /* Add semi-transparent background for readability */
    color: white;
}

h1, h2 {
    text-shadow: 2px 2px 8px #000000; /* Apply black outline to the text */
}

/* Responsive font size for h1 (Title) */
h1 {
    font-size: 50px; /* Default font size for larger screens */
}

@media screen and (max-width: 768px) {
    h1 {
        font-size: 35px; /* Smaller font size for medium screens (e.g., tablets) */
    }
}

@media screen and (max-width: 480px) {
    h1 {
        font-size: 30px; /* Even smaller font size for smaller screens (e.g., mobile phones) */
    }
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title and heading styling
st.markdown(
    "<h1 style='text-align: center; color: black;'>🏏 IPL Win Predictor 🏏</h1>",
    unsafe_allow_html=True
)

# Dropdown options
teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load the trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))
# Input section with styling
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox(
        'Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox(
        'Select the Bowling Team', sorted(teams))

selected_city = st.selectbox('Select Host City', sorted(cities))

target = st.number_input('Enter Target Score', min_value=1, step=1)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Enter Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input(
        'Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Out', min_value=0, max_value=10, step=1)

# Predict probability on button click
if st.button('Predict Probability'):
    # Calculate derived features
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Create input DataFrame
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

    # Predict probabilities
    result = pipe.predict_proba(input_df)
    loss_prob = result[0][0]
    win_prob = result[0][1]

    # Display results beautifully
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 30px;">
            <h2 style="color: #FFD700; text-shadow: 2px 2px 5px #000000;">{batting_team}: {win_prob * 100:.2f}%</h2>
            <h2 style="color: #FF6347; text-shadow: 2px 2px 5px #000000;">{bowling_team}: {loss_prob * 100:.2f}%</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
