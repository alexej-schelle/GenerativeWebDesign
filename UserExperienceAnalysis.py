# Header Definition of Source Data and Models
import pandas as pd
import numpy as np

from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Defition of Data Set Path
df = pd.read_csv("/Users/krealix/Desktop/IU_Internationale_Hochschule/SoSe2024/DSUE042301_VC_SoSe_2024/PythonSource/UserExperienceData.csv")

# Einsicht in die Hauptstruktur im Datenfile
print(df.head())

# Information über die Datenstruktur
print(df.info())

# Beschreibung der Datenstruktur
print(df.describe())

# Datensatz für Alter ist als DataFrame in df gespeichert und wird in unteschiedliche Intervalle segmentiert
age_bins = [0, 18, 30, 40, 50, 60]
age_group_dummies = pd.get_dummies(pd.cut(df['Age'], bins=age_bins, labels=['0-18', '19-30', '31-40', '41-50', '51-60'], right=False), prefix='Age')
age_group_dummies = age_group_dummies.astype(int)

# Zusammensetzen der Segemente mit unterschiedlichen Alters-Intervallen zu einem Datensatz (nach Altersintervallen strukturiert)
df = pd.concat([df, age_group_dummies], axis=1)

# GenderType nochmals als Boolscher Wert definiert
gender_dummies = pd.get_dummies(df['Gender'])
gender_dummies = gender_dummies.astype(int)
df = pd.concat([df, gender_dummies], axis=1)

# VRHeadset nochmals als Boolscher Wert definiert
vrheadset_dummies = pd.get_dummies(df['VRHeadset'])
vrheadset_dummies = vrheadset_dummies.astype(int)
df = pd.concat([df, vrheadset_dummies], axis=1)

# Datensatz für Dauer ist als DataFrame in df gespeichert und wird in unteschiedliche Intervalle segmentiert
duration_bins = [5, 15, 25, 35, 45, 55, 65]
duration_group_dummies = pd.get_dummies(pd.cut(df['Duration'], bins=duration_bins, labels=['5-15', '15-25', '25-35', '35-45', '45-55', '55-65'], right=False), prefix='Duration')
duration_group_dummies = duration_group_dummies .astype(int)
df = pd.concat([df, duration_group_dummies], axis=1)

# Datensatz für MootionSickness ist als DataFrame in df gespeichert und wird in unteschiedliche Intervalle segmentiert
ms_dummies = pd.get_dummies(df['MotionSickness'], prefix='MotionSickness_level')
ms_dummies = ms_dummies.astype(int)
df = pd.concat([df, ms_dummies], axis=1)

# del df['Gender']
# del df['VRHeadset']
# del df['MotionSickness']

# Erstellung eines ML-Modells
A_input = df['ImmersionLevel']
A_output = df.drop('ImmersionLevel', axis=1)

# Data_train --> Daten, welche zum Training verwendet werden (Input) --> Beschreibenden Daten 
# Data_test --> Daten, welche zum Testen verwendet werden (Input) --> Beschreibenden Daten

# data_train --> Daten, welche zum Training verwendet werden (Output) --> Ergebniswerte
# data_test --> Daten, welche zum Testen verwendet werden (Output) --> Ergebniswerte

Data_train, Data_test, data_train, data_test = train_test_split(A_input, A_output, test_size=0.2, random_state=42)

def model_evaluation(data_pred, y_test=data_test):
    
    #Mean Absolute Error
    print('*'*70)
    print(f'Mean Absolute Error: {mean_absolute_error(data_pred, y_test)}')
    print('*'*70)
    
    #accuracy score of the given Metrics
    print(f'Accuracy Score: {accuracy_score(data_pred, y_test)}')
    print('*'*70)
    
    #Confusion Matrix
    print(f"Confusion Matrix: \n{confusion_matrix(data_pred, y_test)}")
    print('*'*70)
    
    #classification report 
    print(f"Classification Report: \n{classification_report(data_pred, y_test)}")

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(Data_train, data_train)

# Data_train --> 2D
# data_tain --> 1D