import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
from sklearn.preprocessing import StandardScaler

# Load model and data
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'test.csv')

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Grade mapping
grade_map = {
    4: 'A (≥3.5)',
    3: 'B (3.0-3.5)',
    2: 'C (2.5-3.0)',
    1: 'D (1.0-2.5)',
    0: 'F (<1.0)'
}

# Calculate scaling parameters from the test data (assuming it represents the training distribution)
scaler = StandardScaler()
numerical_cols = ['Age', 'StudyTimeWeekly', 'Absences', 'TotalActivities']
scaler.fit(df[numerical_cols])

X_test = df.drop(['GradeClass'], axis=1)
y_test = df['GradeClass']

# Predict on full dataset
def get_model_preds(data):
    drop_cols = [col for col in ['StudentID', 'GPA', 'GradeClass'] if col in data.columns]
    X = data.drop(columns=drop_cols)

    y = data['GradeClass'] if 'GradeClass' in data.columns else [None] * len(data)
    preds = model.predict(X)

    return pd.DataFrame({
        'StudentID': data['StudentID'] if 'StudentID' in data.columns else range(len(data)),
        'Actual': [grade_map[val] if val is not None else "N/A" for val in y],
        'Predicted': preds,
        'PredictedLabel': pd.Series(preds).map(grade_map)
    })

# Prediction helper
def predict_grade_from_input(user_inputs):
    # Process numerical inputs - scale them using the same scaler as training data
    numerical_values = np.array([
        user_inputs['age'],
        user_inputs['study'],
        user_inputs['absences'],
        0  # Placeholder for TotalActivities (will be calculated)
    ]).reshape(1, -1)
    
    scaled_values = scaler.transform(numerical_values)

    # Create input dictionary with scaled values
    input_dict = {
        'Age': scaled_values[0][0],
        'StudyTimeWeekly': scaled_values[0][1],
        'Absences': scaled_values[0][2],
        'Tutoring': user_inputs['tutoring'],
        'Extracurricular': user_inputs['extra'],
        'Sports': user_inputs['sports'],
        'Music': user_inputs['music'],
        'Volunteering': user_inputs['volunteer'],
    }

    # Add all encoded categorical features (default to False)
    categorical_flags = {
        'Gender_1': False,
        'Ethnicity_1': False,
        'Ethnicity_2': False,
        'Ethnicity_3': False,
        'ParentalEducation_1': False,
        'ParentalEducation_2': False,
        'ParentalEducation_3': False,
        'ParentalEducation_4': False,
        'ParentalSupport_1': False,
        'ParentalSupport_2': False,
        'ParentalSupport_3': False,
        'ParentalSupport_4': False,
    }

    # Calculate total activities
    total_activities = (
        user_inputs['tutoring'] +
        user_inputs['extra'] +
        user_inputs['sports'] +
        user_inputs['music'] +
        user_inputs['volunteer']
    )
    
    # Scale total activities
    scaled_activities = scaler.transform([[0, 0, 0, total_activities]])[0][3]
    input_dict['TotalActivities'] = scaled_activities

    # Merge everything
    full_input = {**input_dict, **categorical_flags}

    # Reorder to match training feature order
    expected_order = [
        'Age', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'Extracurricular',
        'Sports', 'Music', 'Volunteering', 'TotalActivities', 'Gender_1',
        'Ethnicity_1', 'Ethnicity_2', 'Ethnicity_3', 'ParentalEducation_1',
        'ParentalEducation_2', 'ParentalEducation_3', 'ParentalEducation_4',
        'ParentalSupport_1', 'ParentalSupport_2', 'ParentalSupport_3', 'ParentalSupport_4'
    ]

    # Create input dataframe
    input_df = pd.DataFrame([full_input])
    input_df = input_df[expected_order]

    # Ensure boolean features are typed correctly
    bool_cols = [col for col in input_df.columns if input_df[col].dtype == 'bool']
    input_df[bool_cols] = input_df[bool_cols].astype(int)  # Convert to 0/1

    # Run prediction
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][pred]

    return f"Predicted Grade: {grade_map[pred]} (Confidence: {prob:.1%})"

# Dash app
app = dash.Dash(__name__)
server = app.server

# Custom CSS styles
styles = {
    'container': {
        'fontFamily': 'Arial, sans-serif',
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '20px'
    },
    'header': {
        'textAlign': 'center',
        'color': '#2c3e50',
        'marginBottom': '30px'
    },
    'formSection': {
        'backgroundColor': '#f8f9fa',
        'borderRadius': '8px',
        'padding': '20px',
        'marginBottom': '30px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    },
    'formTitle': {
        'color': '#2c3e50',
        'marginBottom': '20px',
        'borderBottom': '1px solid #eee',
        'paddingBottom': '10px'
    },
    'formGroup': {
        'marginBottom': '15px'
    },
    'formLabel': {
        'display': 'block',
        'marginBottom': '5px',
        'fontWeight': '600',
        'color': '#495057'
    },
    'formInput': {
        'width': '100%',
        'padding': '8px 12px',
        'border': '1px solid #ced4da',
        'borderRadius': '4px',
        'fontSize': '16px'
    },
    'formButton': {
        'backgroundColor': '#007bff',
        'color': 'white',
        'border': 'none',
        'padding': '10px 20px',
        'borderRadius': '4px',
        'cursor': 'pointer',
        'fontSize': '16px',
        'marginTop': '10px'
    },
    'formButtonHover': {
        'backgroundColor': '#0069d9'
    },
    'predictionOutput': {
        'marginTop': '20px',
        'padding': '15px',
        'backgroundColor': '#e2f3ff',
        'borderRadius': '4px',
        'fontSize': '18px',
        'fontWeight': '600',
        'color': '#004085'
    },
    'graphSection': {
        'marginTop': '30px'
    }
}

app.layout = html.Div(style=styles['container'], children=[
    html.H1("Student Grade Predictor", style=styles['header']),
    
    # Form Section
    html.Div(style=styles['formSection'], children=[
        html.H3("Enter Student Information", style=styles['formTitle']),
        
        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '20px'}, children=[
            # Column 1
            html.Div(children=[
                html.Div(style=styles['formGroup'], children=[
                    html.Label("Age", style=styles['formLabel']),
                    dcc.Input(id='age', type='number', value=17, min=10, max=25, style=styles['formInput'])
                ]),
                html.Div(style=styles['formGroup'], children=[
                    html.Label("Study Time Weekly (hours)", style=styles['formLabel']),
                    dcc.Input(id='study', type='number', value=10, min=0, max=50, style=styles['formInput'])
                ]),
                html.Div(style=styles['formGroup'], children=[
                    html.Label("Absences", style=styles['formLabel']),
                    dcc.Input(id='absences', type='number', value=5, min=0, max=100, style=styles['formInput'])
                ]),
            ]),
            
            # Column 2
            html.Div(children=[
                html.Div(style=styles['formGroup'], children=[
                    html.Label("Tutoring", style=styles['formLabel']),
                    dcc.Dropdown(
                        id='tutoring',
                        options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
                        value=1,
                        clearable=False,
                        style={**styles['formInput'], 'padding': '0'}
                    )
                ]),
                html.Div(style=styles['formGroup'], children=[
                    html.Label("Extracurricular Activities", style=styles['formLabel']),
                    dcc.Dropdown(
                        id='extra',
                        options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
                        value=1,
                        clearable=False,
                        style={**styles['formInput'], 'padding': '0'}
                    )
                ]),
                html.Div(style=styles['formGroup'], children=[
                    html.Label("Sports", style=styles['formLabel']),
                    dcc.Dropdown(
                        id='sports',
                        options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
                        value=0,
                        clearable=False,
                        style={**styles['formInput'], 'padding': '0'}
                    )
                ]),
                html.Div(style=styles['formGroup'], children=[
                    html.Label("Music", style=styles['formLabel']),
                    dcc.Dropdown(
                        id='music',
                        options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
                        value=0,
                        clearable=False,
                        style={**styles['formInput'], 'padding': '0'}
                    )
                ]),
                html.Div(style=styles['formGroup'], children=[
                    html.Label("Volunteering", style=styles['formLabel']),
                    dcc.Dropdown(
                        id='volunteer',
                        options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}],
                        value=1,
                        clearable=False,
                        style={**styles['formInput'], 'padding': '0'}
                    )
                ]),
            ])
        ]),
        
        html.Div(style={'textAlign': 'center', 'marginTop': '20px'}, children=[
            html.Button("Predict Grade", id='predict-btn', n_clicks=0, style=styles['formButton'])
        ]),
        
        html.Div(id='prediction-output', style=styles['predictionOutput'])
    ]),
    
    # Graph Section
    html.Div(style=styles['graphSection'], children=[
        html.H3("Actual vs Predicted Grades (Test Data)"),
        dcc.Graph(id='actual-vs-predicted')
    ])
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    Input('age', 'value'),
    Input('study', 'value'),
    Input('absences', 'value'),
    Input('tutoring', 'value'),
    Input('extra', 'value'),
    Input('sports', 'value'),
    Input('music', 'value'),
    Input('volunteer', 'value')
)
def handle_prediction(n_clicks, age, study, absences, tutoring, extra, sports, music, volunteer):
    if n_clicks == 0:
        return ""
    
    # Validate inputs
    if None in [age, study, absences, tutoring, extra, sports, music, volunteer]:
        return "Please fill in all fields"
    
    input_data = {
        'age': age,
        'study': study,
        'absences': absences,
        'tutoring': tutoring,
        'extra': extra,
        'sports': sports,
        'music': music,
        'volunteer': volunteer
    }
    return predict_grade_from_input(input_data)

@app.callback(
    Output('actual-vs-predicted', 'figure'),
    Input('predict-btn', 'n_clicks')
)
def update_chart(n):
    results = get_model_preds(df.copy())
    fig = px.scatter(results, x='Actual', y='PredictedLabel',
                     hover_data=['StudentID'],
                     title="Actual vs Predicted Grade Class")
    fig.update_layout(
        xaxis_title="Actual Grade",
        yaxis_title="Predicted Grade",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#495057')
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))