import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def main():
    st.title("COVID-19 SYMPTOMPS PREDICTOR WEB APP")
    st.text("")
    st.sidebar.title("COVID-19 SYMPTOMPS PREDICTOR Web App")
    

    @st.cache(persist=True)
    def load_data():
        df = pd.read_csv("COVID-19dataset.csv")
        return df
    
   
    @st.cache(persist=True)
    def split(df):
        y = df['Corona result']
        x = df.drop(columns=['Corona result'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    

    df = load_data()
    
    if st.checkbox("Show raw data", False):
        st.subheader("COVID-19 SYMPTOMPS Data Set")
        st.write(df)
    
        
    x_train, x_test, y_train, y_test = split(df)
    
    st.text("Please fill the details in the sidebar.")

    Age = st.sidebar.slider("Enter your Age", 0, 100, key='age')
    
    Gender = st.sidebar.radio("Select your Gender", ("male", "female"), key='Gender')
    
    bodytemperature = st.sidebar.number_input("Enter your bodytemperature in farenheit ?", 0.0 , 150.0, step=0.1, key='bodytemperature')
    
    dry_cough = st.sidebar.radio("Do you have dry cough ?", ("yes", "no"), key='dry_cough')   
    
    sour_throat = st.sidebar.radio("Do you have sour throat ?", ("yes", "no"), key='sour_throat')
    
    weakness = st.sidebar.radio("Are you having weakness ?", ("yes", "no"), key='weakness')
    
    breathing_difficulty = st.sidebar.radio("Are you having difficulty in breathing ?", ("yes", "no"), key='breathing_difficulty')
    
    drowsiness = st.sidebar.radio("Are you feeling drowsiness ?", ("yes", "no"), key='drowsiness')
    
    pain_in_chest = st.sidebar.radio("Do you feel pain in chest ?", ("yes", "no"), key='pain_in_chest')
    
    travelToinfectedCountry = st.sidebar.radio("Have you recently traveled to any infected country ?", ("yes", "no"), key='travelToinfectedCountry')
    
    diabities = st.sidebar.radio("Do you have diabities ?", ("yes", "no"), key='diabities')
    
    heart_disease = st.sidebar.radio("Do you have heart disease ?", ("yes", "no"), key='heart_disease')
    
    lung_disease = st.sidebar.radio("Do you have lung disease ?", ("yes", "no"), key='lung_disease')
    
    stroke = st.sidebar.radio("Did you have any stroke ?", ("yes", "no"), key='stroke')
    
    symptompts_progressed = st.sidebar.radio("Have the symptompts progressed ?", ("yes", "no"), key='symptompts_progressed')
    
    high_bp = st.sidebar.radio("Do yo have high bp ?", ("yes", "no"), key='high_bp')
    
    kidney_disease = st.sidebar.radio("Do you have kidney disease ?", ("yes", "no"), key='kidney_disease')
    
    changeINappetite = st.sidebar.radio("Was there any change in appetite ?", ("yes", "no"), key='changeINappetite')
    
    lossINsmell = st.sidebar.radio("Have you lost any sense of smell ?", ("yes", "no"), key='lossINsmell')


    if Gender =='male':
        Gender = 1
    else:
        Gender = 0

    if dry_cough =='yes':
        dry_cough = 1
    else:
        dry_cough = 0
    
    if sour_throat =='yes':
        sour_throat = 1
    else:
        sour_throat = 0
    
    if weakness =='yes':
        weakness = 1
    else:
        weakness = 0
    
    if breathing_difficulty =='yes':
        breathing_difficulty = 1
    else:
        breathing_difficulty = 0
    
    if drowsiness =='yes':
        drowsiness = 1
    else:
        drowsiness = 0
    
    if pain_in_chest =='yes':
        pain_in_chest = 1
    else:
        pain_in_chest = 0
    
    if travelToinfectedCountry =='yes':
        travelToinfectedCountry = 1
    else:
        travelToinfectedCountry = 0
    
    if diabities =='yes':
        diabities = 1
    else:
        diabities = 0
    
    if heart_disease =='yes':
        heart_disease = 1
    else:
        heart_disease = 0
    
    if lung_disease =='yes':
        lung_disease = 1
    else:
        lung_disease = 0
    
    if stroke =='yes':
        stroke = 1
    else:
        stroke = 0
    
    if symptompts_progressed =='yes':
        symptompts_progressed = 1
    else:
        symptompts_progressed = 0
    
    if high_bp =='yes':
        high_bp = 1
    else:
        high_bp = 0
    
    if kidney_disease =='yes':
        kidney_disease = 1
    else:
        kidney_disease = 0
    
    if changeINappetite =='yes':
        changeINappetite = 1
    else:
        changeINappetite = 0
    
    if lossINsmell =='yes':
        lossINsmell = 1
    else:
        lossINsmell = 0
    

    model = RandomForestRegressor(n_estimators=100, max_features=0.7, bootstrap=True, max_depth=10, min_samples_leaf=5, random_state=42)
    model.fit(x_train, y_train)

    x = model.predict([[Age,Gender,bodytemperature,dry_cough,sour_throat,weakness,breathing_difficulty,drowsiness,pain_in_chest,travelToinfectedCountry,diabities,heart_disease,lung_disease,stroke,symptompts_progressed,high_bp,kidney_disease,changeINappetite,lossINsmell]])
    predictions = model.predict(x_test)


    if st.checkbox("Show model r2 score", False):
        st.write("r2 score: ", r2_score(y_test, predictions).round(2))

    if st.button('Click to predict'):
        if x>=0.5:
            st.error("You may be infected. Please go see a doctor.")
        else :
            st.success("You need not worry, you are not infected. But please follow preventive measures.")

    
if __name__ == '__main__':
    main()
