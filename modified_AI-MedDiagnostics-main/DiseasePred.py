import pickle
import pandas as pd


def recommends_medicine(predicted_disease):
    # Load the disease-medicine dataset
    medicine_df = pd.read_csv("Disease_Medicine.csv")

    # Filter medicines for the predicted disease
    recommended_meds = medicine_df[medicine_df['Disease'] == predicted_disease].iloc[:, 1:].values.flatten()
    return recommended_meds


def predicts(inputs):
    header = ['bloody_stools', 'fecal_leakage', 'swelling', 'dizziness', 'confusion', 'fatigue', 'itching', 'vomiting', 'arm_pain',
              'cough', 'muscle_pain', 'depression', 'fever', 'painful_bowel_moments', 'urine_blood', 'sweating', 'nausea',
              'stiff_neck', 'decreased_appetite', 'weak', 'wheezing', 'bleeding', 'hives', 'bleed', 'headache', 'dry_mouth', 'sweat',
              'stomach_pain', 'stool_pressure', 'anxiety', 'shoulder_pain', 'anus_itching', 'vision_problem', 'abdominal_pain',
              'chest_pain', 'weight_loss', 'diarrhea', 'breath_problems', 'thirsty', 'anus_swelling', 'blood_o_tissue', 'constipation',
              'neck_pain', 'low_heartbeat', 'more_urine', 'low_breath', 'muscle_cramps', 'muscle_spasm', 'yawning', 'rash', 'back_pain',
              'anal_bleeding', 'lump_anus', 'cold', 'skin_rash', 'neck_stiff']

    # Load disease symptoms dataset
    df = pd.read_csv("Disease_Symptoms.csv")
    disease = set(df.iloc[:, 0])
    disease = list(disease)
    disease.sort()

    model_inputs5 = [0] * len(header)

    # Mark input symptoms
    for element in range(len(header)):
        if header[element] in inputs:
            model_inputs5[element] = 1

    # Load pre-trained decision tree model
    with open("DiseasePrediction(Dec)", "rb") as f:
        Model_Decision_Tree = pickle.load(f)

    # Predict disease based on symptoms
    pred = Model_Decision_Tree.predict([model_inputs5])
    predicted_disease = disease[pred[0]]

    # Fetch recommended medicines for the predicted disease
    recommended_meds = recommends_medicine(predicted_disease)

    return predicted_disease, recommended_meds


if __name__ == '__main__':
    # Example: Testing the prediction function with symptoms
    symptoms = ['fever', 'vomiting', 'headache', 'sweating']
    disease, medicines = predicts(symptoms)

    # HTML output (in terminal for now)
    print(f"<h2>Predicted Disease: {disease}</h2>")
    print("<h3>Recommended Medicines:</h3>")
    for med in medicines:
        print(f"<p>{med}</p>")
