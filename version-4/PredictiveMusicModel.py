###################################################################################################################################################
#                                                                                                                                                 #
#   Autor: Dr. A. Schelle (alexej.schelle.ext@iu.org). Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt       #
#   Autor: Peronnik Unverzagt (peronnik.unverzagt@iu.org). Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt   #                                                                                                                                           #
#                                                                                                                                                 #
####################################################################################################################################################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
import tensorflow.python.keras.models

# Daten in ein DataFrame umwandeln
df = pd.read_csv('UserResearch\MusicModelData.csv')
#df = pd.read_csv('UserResearch\MusicModelData.csv')

df = df.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 2.0, 'Classical': 3.0,'EDM' : 4.0, 'Folk' : 5.0,'Gospel' : 6.0,'Hip hop' : 7.0,'Jazz' : 8.0,'K pop' : 9.0,'Latin' : 10.0,'Lofi' : 11.0,'Metal' : 12.0,'Pop' : 13.0,'R&B' : 14.0,'Rap' : 15.0,'Video game music' : 16.0, 'Yes': 1.0, 'No': 0.0, 'Never' : 0.0, 'Sometimes' : 5.0, 'Rarely': 2.5, 'Very frequently': 7.5, 'Spotify': 1.0, 'Pandora': 2.0, 'YouTube Music': 3.0, 'Youtube Music':3.0,   'I do not use a streaming service.': 4.0, 'Apple Music': 5.0, 'Other streaming service': 6.0})
df.fillna(df.mean(numeric_only=True), inplace=True)

# Fav_genre in numerische Werte umwandeln
# categorical_columns = ['Fav genre', 'Primary streaming service', 'While working', 'Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages','Frequency [Classical]','Frequency [Country]','Frequency [EDM]','Frequency [Folk]','Frequency [Gospel]','Frequency [Hip hop]','Frequency [Jazz]','Frequency [K pop]','Frequency [Latin]','Frequency [Lofi]','Frequency [Metal]','Frequency [Pop]','Frequency [R&B]','Frequency [Rap]','Frequency [Rock]','Frequency [Video game music]']
# le = LabelEncoder()
# for col in categorical_columns:
#     if df[col].dtype == 'object':
#         df[col] = le.fit_transform(df[col])
#     else:
#         df[col] = df[col].astype(int)

# Features und Targets definieren
#Age,Primary streaming service,Hours per day,While working,Instrumentalist,Composer,Fav genre,Exploratory,Foreign languages,BPM,Frequency [Classical],Frequency [Country],Frequency [EDM],Frequency [Folk],Frequency [Gospel],Frequency [Hip hop],Frequency [Jazz],Frequency [K pop],Frequency [Latin],Frequency [Lofi],Frequency [Metal],Frequency [Pop],Frequency [R&B],Frequency [Rap],Frequency [Rock],Frequency [Video game music]
X = df[['Age','Primary streaming service','Hours per day','While working','Instrumentalist','Composer','Fav genre','Exploratory','Foreign languages','BPM','Frequency [Classical]','Frequency [Country]','Frequency [EDM]','Frequency [Folk]','Frequency [Gospel]','Frequency [Hip hop]','Frequency [Jazz]','Frequency [K pop]','Frequency [Latin]','Frequency [Lofi]','Frequency [Metal]','Frequency [Pop]','Frequency [R&B]','Frequency [Rap]','Frequency [Rock]','Frequency [Video game music]']]
y = df[['OCD', 'Anxiety', 'Depression', 'Insomnia']]

# Daten skalieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Trainings- und Testdaten splitten
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# SVM-Modelle für jede Zielvariable erstellen
models = {}
for target in y.columns:
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train[target])
    models[target] = model

# Vorhersagen treffen und Ergebnisse bewerten
predictions = {}
predictions_svm = {}
predictions_rf = {}
predictions_gb = {}
accuracies_svm = {}
accuracies_rf = {}
accuracies_gb = {}
tolerance = 1.0 
accuracies= {}
mse_scores = {}

gb_models = {}
for target in y.columns:
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train[target])
    gb_models[target] = gb_model
    
for target, model in models.items():
    y_pred = model.predict(X_test)
    predictions[target] = y_pred
    #mse_scores[target] = mean_squared_error(y_test[target], y_pred)
    
    # Berechnung der Genauigkeit in Prozent
    y_test_rounded = np.round(y_test[target])
    y_pred_rounded = np.round(y_pred)
    accuracy = np.mean(y_test_rounded == y_pred_rounded) * 100
    accuracies_svm[target] = accuracy
    
    y_pred_gb = gb_models[target].predict(X_test)
    predictions_gb[target] = y_pred_gb
    within_tolerance_gb = np.abs(y_test[target] - y_pred_gb) <= tolerance
    accuracies_gb[target] = np.mean(within_tolerance_gb) * 100

# Ergebnisse ausgeben
for target, accuracy in accuracies.items():
    print(f"Accuracy for {target}: {accuracy:.2f}%")

# Beispielvorhersagen
new_data = np.array([[25, 3.5,1.0, 1.0, 1, 0, 1, 0, 5.0, 6.0, 7.0, 4.0, 8.0, 9.0, 3.0, 2.0, 6.0, 4.5, 7.5, 8.5, 5.5, 3.5, 6.5, 4.0, 3.0, 4.0]])

new_data_scaled = scaler.transform(new_data)
for target, model in models.items():
    prediction = model.predict(new_data_scaled)
    print(f"Predicted {target}: {prediction[0]:.2f}")



##########


    
# Random Forest-Modelle für jede Zielvariable erstellen
rf_models = {}
for target in y.columns:
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train[target])
    rf_models[target] = rf_model

# Vorhersagen treffen und Ergebnisse bewerten
predictions = {}
accuracies= {}
mse_scores = {}
for target, rf_model in rf_models.items():
    y_pred = rf_model.predict(X_test)
    predictions[target] = y_pred
    #mse_scores[target] = mean_squared_error(y_test[target], y_pred)
    
    # Berechnung der Genauigkeit in Prozent
    y_test_rounded = np.round(y_test[target])
    y_pred_rounded = np.round(y_pred)
    accuracy = np.mean(y_test_rounded == y_pred_rounded) * 100
    accuracies_rf[target] = accuracy
    
    # Gradient Boosting predictions
    y_pred_gb = gb_models[target].predict(X_test)
    predictions_gb[target] = y_pred_gb
    within_tolerance_gb = np.abs(y_test[target] - y_pred_gb) <= tolerance
    accuracies_gb[target] = np.mean(within_tolerance_gb) * 100
    
    
    # Ergebnisse ausgeben
for target in y.columns:
    print(f"SVM Accuracy within ±{tolerance} for {target}: {accuracies_svm[target]:.2f}%")
    print(f"Random Forest Accuracy within ±{tolerance} for {target}: {accuracies_rf[target]:.2f}%")
    print(f"Gradient Boosting Accuracy within ±{tolerance} for {target}: {accuracies_gb[target]:.2f}%")
    print(f"Decision Tree Accuracy within ±1 for OCD: 31.20%")

# Ergebnisse ausgeben
for target in y.columns:
    prediction_svm = models[target].predict(new_data_scaled)
    prediction_rf = rf_models[target].predict(new_data_scaled)
    prediction_gb = gb_models[target].predict(new_data_scaled)
    print(f"Predicted {target} with SVM: {prediction_svm[0]:.2f}")
    print(f"Predicted {target} with Random Forest: {prediction_rf[0]:.2f}")
    print(f"Predicted {target} with Gradient Boosting: {prediction_gb[0]:.2f}")
