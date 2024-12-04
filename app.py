from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Créer l'application Flask
app  =  Flask(__name__)

# Charger le modèle et le scaler
model  =  joblib.load('random_forest_model.joblib')  # Charger le modèle
scaler  =  joblib.load('scaler.joblib')  # Charger le scaler

@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')  # Afficher le formulaire HTML

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        # Récupérer les données envoyées par le formulaire
        inches  =  float(request.form['Inches'])
        ram  =  float(request.form['Ram'])
        weight  =  float(request.form['Weight'])
        screen_width  =  float(request.form['Screen Width'])
        screen_height  =  float(request.form['Screen Height'])
        frequency  =  float(request.form['Frequency'])
        memory_size  =  float(request.form['Memory Size'])

        # Organiser les données sous forme de DataFrame pandas (même format que lors de l'entraînement)
        data  =  {
            'Inches': [inches],
            'Ram': [ram],
            'Weight': [weight],
            'Screen Width': [screen_width],
            'Screen Height': [screen_height],
            'Frequency': [frequency],
            'Memory Size': [memory_size]
        }
        input_data  =  pd.DataFrame(data)

        # Si la mémoire est en Mo, convertir en Go (si nécessaire)
        if 'Memory Size' in input_data.columns:
            input_data['Memory Size']  =  input_data['Memory Size'] / 1024  # Conversion en Go

        # Appliquer la mise à l'échelle (scaling) aux nouvelles données
        input_data_scaled  =  scaler.transform(input_data)

        # Faire la prédiction avec le modèle
        predicted_price  =  model.predict(input_data_scaled)

        # Afficher la page avec la prédiction
        return render_template('index.html', predicted_price = predicted_price[0])

    except Exception as e:
        # En cas d'erreur, retourner un message d'erreur
        return render_template('index.html', predicted_price = f"Erreur: {str(e)}")

if __name__  ==  '__main__':
    app.run(debug = True)
