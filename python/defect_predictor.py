import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

def enhance_dataset_and_predict():
    data_path = '../data/defect_data.csv'
    
    # Load dataset if it exists, otherwise create a dummy one
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        print("Data file not found. Generating a dummy dataset.")
        np.random.seed(42)
        df = pd.DataFrame({'defect_id': range(1, 1001)})

    # Add synthetic columns required for the predictive model and Tableau dashboards
    np.random.seed(42)
    n = len(df)
    
    # Assembly stations as requested
    stations = ['dial', 'case', 'strap', 'movement']
    df['Assembly_Station'] = np.random.choice(stations, n)
    
    # Shifts and Inspector IDs
    df['Shift'] = np.random.choice([1, 2, 3], n)
    df['Inspector_ID'] = np.random.choice(['I-101', 'I-102', 'I-103', 'I-104'], n)
    
    # Target variable: Defect Present (1/0)
    # Let's say normally it's 1 since it's a defect dataset, but we'll simulate some non-defects
    # or just use severity to derive a 'Defect_Present' probability.
    df['Defect_Present'] = np.random.choice([0, 1], n, p=[0.2, 0.8])
    
    # KPI metrics requested for Tableau (normally 1 unit per row if it's line item)
    # We will simulate production volume metrics for the dashboards
    df['Total_Units_Produced'] = np.random.randint(100, 500, n)
    df['Units_Passed_First_Time'] = (df['Total_Units_Produced'] * np.random.uniform(0.7, 0.98, n)).astype(int)
    df['Units_Reworked'] = (df['Total_Units_Produced'] * np.random.uniform(0.01, 0.15, n)).astype(int)
    df['Units_Rejected'] = df['Total_Units_Produced'] - df['Units_Passed_First_Time'] - df['Units_Reworked']
    
    # Root Cause Analysis (RCA) columns
    rca_mapping = {
        'Loose strap': ('Improper torque', 'Recalibrate tool', 'Set SOP'),
        'Dial scratch': ('Handling error', 'Operator training', 'Visual inspection'),
        'Movement error': ('Calibration issue', 'Reset machine', 'Daily checks'),
        'Case damage': ('Machine fault', 'Replace part', 'Routine maintenance')
    }
    
    # Assign new defect types just for the mapping, or map to existing
    new_defect_types = ['Loose strap', 'Dial scratch', 'Movement error', 'Case damage']
    df['Specific_Defect_Type'] = np.random.choice(new_defect_types, n)
    
    df['Root_Cause'] = df['Specific_Defect_Type'].apply(lambda x: rca_mapping[x][0])
    df['Corrective_Action'] = df['Specific_Defect_Type'].apply(lambda x: rca_mapping[x][1])
    df['Preventive_Action'] = df['Specific_Defect_Type'].apply(lambda x: rca_mapping[x][2])
    
    # CAPA Implementation status
    df['Implementation_Status'] = np.random.choice(['Pending', 'In Progress', 'Completed'], n, p=[0.3, 0.4, 0.3])
    
    # --- ML Predictive Model ---
    # Preparing data
    print("Training Simple Predictive Model...")
    
    # Encoding categorical variables for the model
    X = pd.get_dummies(df[['Assembly_Station', 'Shift', 'Inspector_ID']], drop_first=True)
    y = df['Defect_Present']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict probabilities on the entire dataset
    pred_probabilities = model.predict_proba(X)[:, 1]
    df['Defect_Risk_Probability'] = np.round(pred_probabilities, 4)
    
    print("Model trained. Predictions added.")
    
    # Save the enhanced dataset back to data folder
    enhanced_path = '../data/defect_data_enhanced.csv'
    df.to_csv(enhanced_path, index=False)
    print(f"Enhanced dataset saved to {enhanced_path}")

if __name__ == "__main__":
    enhance_dataset_and_predict()
