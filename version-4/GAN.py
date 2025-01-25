################################################################################################################################################
#                                                                                                                                              #
#   Autor: Dr. A. Schelle (alexej.schelle.ext@iu.org). Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt    #
#                                                                                                                                              #
################################################################################################################################################

# PYTHON ROUTINE zur Modellierung von Immersionslevels durch GAN-Modelle

import os
import sys
import statistics
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





MusicEffects = 'Improve'  # (0 = no effect, 1 = improve , 2 = worse)
Permissions = 'I understand'
Duration = 0.0
StatsClass = "Sometimes"


with open("EstimateDuration\durationEstimate.txt", "r") as file:
    for line in file:
        # Split each line into key and value at ": "
        Duration = line



data = {}
with open("Input.txt", "r") as file:
    for line in file:
        # Split each line into key and value at ": "
        key, value = line.strip().split(": ", 1)
        data[key] = value


# Assign the variables from the data dictionary
Age = int(data["Age"])
Anxiety = int(data["Anxiety Level"])
Depression = int(data["Depression Level"])
Insomnia = int(data["Insomnia Level"])
OCD = int(data["OCD Level"])
Exploratory = data["Exploratory"]
ForeignLanguage = data["Foreign Language"]
FavoriteGenre = data["Favorite Genre"]
Instrumentalist = data["Instrumentalist"]
Composer = data["Composer"]
PrimaryStreamingService = data["Primary Streaming Service"]
WhileWorking = data["While Working"]
MusicEffect = data["Music Effect"]



maximal_accuracy = 95.0 # Maximale Genauigkeit in Prozent

# Open reference file
df = pd.read_csv('MusicExperienceData.csv')

#df_reduced = df[firstParameter].copy()
#age = df_reduced.mean()

accuracy_values = [5.0, 10.0, 25.0, 50.0, 75.0, maximal_accuracy]

counter = 0
for i in range(1, len(df)):
    
    found_value = 0
    similarity = 0
    found_value = 0
    for data_accuracy in accuracy_values:
        if (
            ((df.iloc[i]['Hours per day'] / df['Hours per day'].max()) - float(Duration)) < 0.5
            and (df.iloc[i]['Age'] - float(Age)) / float(Age) * 100.0 < (100.0 - data_accuracy)
            and (df.iloc[i]['Anxiety'] - float(Anxiety)) / float(Anxiety + 1) * 100.0 < (100.0 - data_accuracy)
            and (df.iloc[i]['Depression'] - float(Depression)) / float(Depression + 1) * 100.0 < (100.0 - data_accuracy)
            and (df.iloc[i]['Insomnia'] - float(Insomnia)) / float(Insomnia + 1) * 100.0 < (100.0 - data_accuracy)
            and (df.iloc[i]['OCD'] - float(OCD)) / float(OCD + 1) * 100.0 < (100.0 - data_accuracy)
            and (df.iloc[i]['Music effects'] == MusicEffect)
        ):
            if found_value == 0:
                counter += 1
                found_value = 1

            # Extract genres and other details
            genres = [
                df.iloc[i]['Frequency [Classical]'], df.iloc[i]['Frequency [Country]'],
                df.iloc[i]['Frequency [EDM]'], df.iloc[i]['Frequency [Gospel]'],
                df.iloc[i]['Frequency [Hip hop]'], df.iloc[i]['Frequency [Jazz]'],
                df.iloc[i]['Frequency [K pop]'], df.iloc[i]['Frequency [Folk]'],
                df.iloc[i]['Frequency [Latin]'], df.iloc[i]['Frequency [Lofi]'],
                df.iloc[i]['Frequency [Metal]'], df.iloc[i]['Frequency [Pop]'],
                df.iloc[i]['Frequency [Rap]'], df.iloc[i]['Frequency [Rock]'],
                df.iloc[i]['Frequency [R&B]'], df.iloc[i]['Frequency [Video game music]']
            ]
            duration = df.iloc[i]['Hours per day']
            bpm = df.iloc[i]['BPM']
            break  # Exit accuracy loop once a match is found

       
       

numGenre = genres.count(StatsClass)
print('genres:' , genres)

print('Number of genres: ' , numGenre)
print('Genres : ',genres)
print ('Duration : ' , duration)
print ('BPM: ', bpm)

print('Counter: ' , counter)

# TO DOs: Modellierung der Daten als Funktion der maximalen Genauigkeit (maximal_accuracy)
