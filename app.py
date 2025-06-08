from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
pipe = pickle.load(open("10 pipe.pkl", 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[
        data['Day_of_week'], data['Age_band_of_driver'], data['Sex_of_driver'], data['Educational_level'],
        data['Vehicle_driver_relation'], data['Driving_experience'], data['Type_of_vehicle'], data['Owner_of_vehicle'],
        data['Service_year_of_vehicle'], data['Defect_of_vehicle'], data['Area_accident_occured'], data['Lanes_or_Medians'],
        data['Road_allignment'], data['Types_of_Junction'], data['Road_surface_type'], data['Road_surface_conditions'],
        data['Light_conditions'], data['Weather_conditions'], data['Type_of_collision'], data['Number_of_vehicles_involved'],
        data['Number_of_casualties'], data['Vehicle_movement'], data['Casualty_class'], data['Sex_of_casualty'],
        data['Age_band_of_casualty'], data['Casualty_severity'], data['Work_of_casualty'], data['Fitness_of_casualty'],
        data['Pedestrian_movement'], data['Cause_of_accident'], data['Hour_of_Day']
    ]], dtype=object)
    prediction = pipe.predict(features)[0]
    label = {2: "Slight Injury", 1: "Serious Injury", 0: "Fatal Injury"}[prediction]
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)