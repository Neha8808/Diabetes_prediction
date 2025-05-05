import streamlit as st
import pandas as pd
import joblib 


model=joblib.load('diabetes_pred.pkl')



def main():
    # Set the title and description of the app
  
    st.title("Diabetes Prediction App")
    
    st.write("Please enter the following details:")

    # Create input fields for user data
    pregnancies = st.slider("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.slider("Glucose Level", min_value=0, max_value=200, value=0)
    blood_pressure = st.slider("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=0)
    skin_thickness = st.slider("Skin Thickness (mm)", min_value=0, max_value=100, value=0)
    insulin = st.slider("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=0)
    bmi = st.slider("Body Mass Index (BMI)", min_value=0.0, max_value=50.0, value=0.0)
    diabetes_pedigree_function = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0)
    age = st.slider("Age (years)", min_value=1, max_value=120, value=1)

    # Create a button to trigger prediction
    if st.button("Predict"):
        # Prepare the input data for prediction
        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, diabetes_pedigree_function, age]],
                                   columns=["Pregnancies", "Glucose", "BloodPressure",
                                            "SkinThickness", "Insulin", "BMI",
                                            "DiabetesPedigreeFunction", "Age"])

        # Make prediction using the loaded model
        prediction = model.predict(input_data)[0]

        # Display the prediction result
        if prediction == 1:
            st.success("The model predicts that you have diabetes.")
        else:
            st.success("The model predicts that you do not have diabetes.")     

if __name__ == "__main__":
    main()  