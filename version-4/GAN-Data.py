###################################################################################################################################################
#                                                                                                                                                 #
#   Autor: Dr. A. Schelle (alexej.schelle.ext@iu.org). Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt       #
#   Autor: Peronnik Unverzagt (peronnik.unverzagt@iu.org). Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt   #                                                                                                                                           #
#                                                                                                                                                 #
###################################################################################################################################################

# PYTHON ROUTINE zur Modellierung von Daten durch GAN-Netzwerke #

import os
import sys
import statistics
import math
import random
import pandas as pd
import numpy as np
#from numpy import random
import matplotlib.pyplot as plt

from functools import reduce

def custom_round(value):
    rounded_value = round(value)
    if rounded_value == 0:
        if value > 0:
            return 1
        elif value < 0:
            return -1
    return rounded_value

def Generator(ID, length):

    output_key = [0]*length
    output_key[0] = ID

    for k in range(0, length):

        if (k == 1):
            #Age
            age = random.gauss(25.2, 12.05)
            if (age < 0):
                age = age * -1 
                
            output_key[1] = age
                # Alter im Bereich von zehr bis Hundert Jahre
            #output_key[1] = random.poisson(25, size=5000) # Alter im Bereich von zehr bis Hundert Jahre

        if (k == 2):
            #Streaming service
            choose_streamingService = random.randint(0,450)

            streamingServices = ["Pandora", "Spotify", "Apple Music", "YouTube Music", "I do not use a streaming service."]
            if ((choose_streamingService <= 450) and (choose_streamingService > 94)):
                output_key[2] = streamingServices[1]
            elif ((choose_streamingService <= 94) and (choose_streamingService > 71)):
                output_key[2] = streamingServices[3]
            elif ((choose_streamingService <= 71) and (choose_streamingService > 51)):
                output_key[2] = streamingServices[4]
            elif ((choose_streamingService <= 51) and (choose_streamingService > 50)):
                output_key[2] = streamingServices[2]
            elif ((choose_streamingService <= 50) and (choose_streamingService > 11)):
                output_key[2] = streamingServices[0]
                
        if (k == 3):
            # Hours per day
            output_key[3] = np.random.exponential(7.5,None)

        if (k == 4):
            #While Working
            choose_whileWorking = random.randint(0,3)
            
            if (choose_whileWorking == 0):
                output_key[4] = 'No'
            elif (choose_whileWorking >= 1):
                output_key[4] = 'Yes'
            
            
        if (k == 5):

            choose_Instrumentalist = random.randint(0,2)
            
            if (choose_Instrumentalist == 0):
                output_key[5] = 'No'
            elif (choose_Instrumentalist >= 1):
                output_key[5] = 'Yes'
            
        
        if (k == 6):

            choose_Composer = random.randint(0,7)
            
            if (choose_Composer == 0):
                output_key[6] = 'No'
            elif (choose_Composer >= 1):
                output_key[6] = 'Yes'
            

        if (k == 7):

            choose_favGenre = random.randint(0,188)
            #genres = ["Rock", "Pop", "Metal", "Classical", "Video game music", "EDM", "Hip hop", "R&B", "Folk", "K pop", "Country", "Rap", "Jazz", "Lofi", "Gospel", "Latin"]
            #genres_num = [0.0, 1.0, 2.0, 3.0 , 4.0 , 5.0 , 6.0 , 7.0 , 8.0 , 9.0 , 10.0 , 11.0 , 12.0 ,13.0 ,14.0 ,15.0, 16.0]
            if ((choose_favGenre <= 188) and (choose_favGenre > 114)):
                output_key[7] = 1.0 #Rock
            elif ((choose_favGenre <= 114) and (choose_favGenre > 88)):
                output_key[7] = 13.0 #"Pop"
            elif ((choose_favGenre <= 88) and (choose_favGenre > 53)):
                output_key[7] = 12.0# "Metal"
            elif ((choose_favGenre <= 53) and (choose_favGenre > 44)):
                output_key[7] = 16.0#"Video game music"
            elif ((choose_favGenre <= 44) and (choose_favGenre > 37)):
                output_key[7] = 4.0# "EDM"
            elif ((choose_favGenre <= 37) and (choose_favGenre > 35)):
                output_key[7] = 7.0# "Hip Hop"
            elif ((choose_favGenre <= 35) and (choose_favGenre > 34)):
                output_key[7] = 13.0# "R&B"
            elif ((choose_favGenre <= 34) and (choose_favGenre > 30)):
                output_key[7] = 5.0# "Folk"
            elif ((choose_favGenre <= 30) and (choose_favGenre > 26)):
                output_key[7] = 9.0 # "K pop"
            elif ((choose_favGenre <= 26) and (choose_favGenre > 25)):
                output_key[7] = 2.0 # "Country"
            elif ((choose_favGenre <= 25) and (choose_favGenre > 22)):
                output_key[7] = 15.0 # "Rap"
            elif ((choose_favGenre <= 22) and (choose_favGenre > 20)):
                output_key[7] = 8.0 # "Jazz"
            elif ((choose_favGenre <= 20) and (choose_favGenre > 10)):
                output_key[7] = 11.0# "Lofi"
            elif ((choose_favGenre <= 10) and (choose_favGenre > 6)):
                output_key[7] = 6.0# "Gospel"
            elif ((choose_favGenre <= 6) and (choose_favGenre > 0)):
                output_key[7] = 1.0# "Latin"

        if (k == 8):
            choose_Exploratory = random.randint(0,2)
            
            if (choose_Exploratory == 0):
                output_key[8] = 0.0
            elif (choose_Exploratory >= 1):
                output_key[8] = 1.0
            
        if (k==9):
            choose_foreignLan = random.randint(0,7)
            
            if (choose_foreignLan < 3):
                output_key[9] = 'No'
            elif (choose_foreignLan >= 3):
                output_key[9] = 'Yes'
            
        if (k==10):
            #TODO Average BPM per genre
            if (output_key[7] == 'Rock'):
                output_key[10] = random.gauss(124.05, 31.05)
            elif (output_key[7] == 'Pop'):
                output_key[10] = random.gauss(118.90, 28.23)  
            elif (output_key[7] == 'Metal'):
                output_key[10] = random.gauss(139.10, 42.05) 
            elif (output_key[7] == 'Classical'):
                output_key[10] = random.gauss(70.00, 15.00)
            elif (output_key[7] == 'Video game music'):
                output_key[10] = random.gauss(118.90, 28.23)
            elif (output_key[7] == 'EDM'):
                output_key[10] = random.gauss(128.00, 20.00)
            elif (output_key[7] == 'Hip hop'):
                output_key[10] = random.gauss(85.00, 15.00)
            elif (output_key[7] == 'R&B'):
                output_key[10] = random.gauss(90.00, 10.00)
            elif (output_key[7] == 'Folk'):
                output_key[10] = random.gauss(100.00, 12.00)
            elif (output_key[7] == 'K pop'):
                output_key[10] = random.gauss(120.00, 25.00)
            elif (output_key[7] == 'Country'):
                output_key[10] = random.gauss(100.00, 10.00)
            elif (output_key[7] == 'Rap'):
                output_key[10] = random.gauss(85.00, 12.00)
            elif (output_key[7] == 'Jazz'):
                output_key[10] = random.gauss(115.00, 20.00)
            elif (output_key[7] == 'Lofi'):
                output_key[10] = random.gauss(70.00, 10.00)
            elif (output_key[7] == 'Gospel'):
                output_key[10] = random.gauss(100.00, 15.00)
            elif (output_key[7] == 'Latin'):
                output_key[10] = random.gauss(120.00, 20.00)
                
        if (k >= 11 and k < 26):
                choose_Freq = random.randint(0,3)
                frequencies = ["Rarely", "Never", "Sometimes", "Very frequently"]
                for i in range(0,4):
                    if (choose_Freq == i):
                        output_key[k] = frequencies[i]
        
        if (k == 26):
            #OCD
             numbers = [0.0, 2.0, 1.0, 3.0, 5.0, 4.0, 7.0, 6.0, 8.0, 10.0, 9.0]
             weights = [248, 96, 95, 64, 54, 48, 34, 33, 28, 20, 14]

    
             total = sum(weights)
             probabilities = [weight / total for weight in weights]
             
             output_key[26] = random.choices(numbers,probabilities)[0]
        
        if (k == 27):
            #Anxiety
            numbers = [7.0, 8.0, 6.0, 3.0, 10.0, 5.0, 9.0, 4.0, 2.0, 0.0, 1.0]
            weights = [122, 115, 83, 69, 67, 59, 56, 56, 44, 35, 29]

    
            total = sum(weights)
            probabilities = [weight / total for weight in weights]
            
            output_key[27] = random.choices(numbers,probabilities)[0]
           
        
        if (k == 28):
            #Insomnia
            numbers = [0.0, 2.0, 1.0, 3.0, 6.0, 7.0, 4.0, 5.0, 8.0, 10.0, 9.0]
            weights = [149, 88, 82, 68, 62, 59, 59, 58, 49, 34, 27]
            
            total = sum(weights)
            probabilities = [weight / total for weight in weights]
            
            output_key[28] = random.choices(numbers,probabilities)[0]
            
        if (k == 29):
            #Depression
            numbers = [7.0, 2.0, 6.0, 0.0, 8.0, 3.0, 4.0, 5.0, 10.0, 1.0, 9.0]
            weights = [96, 93, 88, 84, 77, 59, 58, 56, 45, 40, 38]

            total = sum(weights)
            probabilities = [weight / total for weight in weights]
            
            output_key[29] = random.choices(numbers,probabilities)[0]
            
        if (k == 30):
            #TODO approximieren je nach Krankheitsbild (4 Verteilungen)
            effects = [3.0, 2.0, 1.0]
            weights = [542, 17, 169]
            
            total = sum(weights)
            probabilities = [weight / total for weight in weights]
            
            output_key[30] = random.choices(effects, probabilities)[0]
            
            #chooose_Effects = random.randint(0,2)
            #if (chooose_Effects == 0):
            #    output_key[30] = 0.0
            #if (chooose_Effects == 1):
            #    output_key[30] = 1.0
            #if (chooose_Effects == 2):
            #    output_key[30] = 2.0
            
            
        if (k == 31):
            output_key[31] = 'I understand.'

    return(output_key)


def Differentiator(df, data_1, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9, accuracy):
#################

#Discriminator parameters:
#   Age     
#   Hours per day
#   Fav genre
#   Exploratory
#   Frequency of fav genre
#   OCD, Anxiety, Insomnia, Depression
#   Music effects
#       If OCD classified : Frequency of Fav Genre not high classified
#       Otherwise Frequency of Fav genre high classified

#    Discriminator 

#################
    
    df = df.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 2.0, 'Classical': 3.0,'EDM' : 4.0, 'Folk' : 5.0,'Gospel' : 6.0,'Hip hop' : 7.0,'Jazz' : 8.0,'K pop' : 9.0,'Latin' : 10.0,'Lofi' : 11.0,'Metal' : 12.0,'Pop' : 13.0,'R&B' : 14.0,'Rap' : 15.0,'Video game music' : 16.0, 'Yes': 1.0, 'No': 0.0})

    similarity_measure = 0
    for j in range(0,736):
         
        if ((df.iloc[j]['Age']-data_1)/data_1*100.0 < (100.0 -  accuracy) and (df.iloc[j]['Exploratory']-data_2)/data_2*100.0 < (100.0 -  accuracy) and (df.iloc[j]['Hours per day']-data_3)/data_3*100.0 < (100.0 -  accuracy) and (df.iloc[j]['Fav genre']-data_4)/data_4*100.0 < (100.0 -  accuracy) and (df.iloc[j]['Music effects']-data_5)/data_5*100.0 < (100.0 -  accuracy) and (df.iloc[j]['OCD']-data_6)/data_6*100.0 < (100.0 -  accuracy) and (df.iloc[j]['Anxiety']-data_7)/data_7*100.0 < (100.0 -  accuracy) and (df.iloc[j]['Insomnia']-data_8)/data_8*100.0 < (100.0 -  accuracy) and (df.iloc[j]['Depression']-data_9)/data_9*100.0 < (100.0 -  accuracy) ):

            similarity_measure = 1
            break
    return similarity_measure


    

data = Generator(0,32)
    # Versuche den Unterschied zwischen dem SVM-Modell und einem Modell wie Decision Tree anhand der Ergebnisse und der Modell-Definitionen zu verstehen
df = pd.read_csv('MusicExperienceData.csv')
print(data)
#print(Differentiator(df, data[1], data[8], data[3], data[7], data[30], data[26], data[27] , data[28], data[29], 10))

# Code for printing to a file
sample = open('UserResearch\MusicModelData.csv', 'w')

# Code for printing to a file
sample_realistic = open('UserResearch\RealisticMusicModelData.csv', 'w')

#data = []
index = 0
sample_size = 2500
data_dimension = 32
data_similarity = 50.0

print(','.join(['Index','Age','Primary streaming service','Hours per day','While working','Instrumentalist','Composer','Fav genre','Exploratory','Foreign languages','BPM','Frequency [Classical]','Frequency [Country]','Frequency [EDM]','Frequency [Folk]','Frequency [Gospel]','Frequency [Hip hop]','Frequency [Jazz]','Frequency [K pop]','Frequency [Latin]','Frequency [Lofi]','Frequency [Metal]','Frequency [Pop]','Frequency [R&B]','Frequency [Rap]','Frequency [Rock]','Frequency [Video game music]','Anxiety','Depression','Insomnia','OCD','Music effects','Permissions']), file=sample)
print(','.join(['Index','Age','Primary streaming service','Hours per day','While working','Instrumentalist','Composer','Fav genre','Exploratory','Foreign languages','BPM','Frequency [Classical]','Frequency [Country]','Frequency [EDM]','Frequency [Folk]','Frequency [Gospel]','Frequency [Hip hop]','Frequency [Jazz]','Frequency [K pop]','Frequency [Latin]','Frequency [Lofi]','Frequency [Metal]','Frequency [Pop]','Frequency [R&B]','Frequency [Rap]','Frequency [Rock]','Frequency [Video game music]','Anxiety','Depression','Insomnia','OCD','Music effects','Permissions']), file=sample_realistic)

for i in range(1, sample_size):
    print (i)
    data = Generator(i, data_dimension)
    print(f"{i},{data[1]},{data[2]},{data[3]},{data[4]},{data[5]},{data[6]},{data[7]},{data[8]},{data[9]},{data[10]},{data[11]},{data[12]},{data[13]},{data[14]},{data[15]},{data[16]},{data[17]},{data[18]},{data[19]},{data[20]},{data[21]},{data[22]},{data[23]},{data[24]},{data[25]},{data[26]},{data[27]},{data[28]},{data[29]},{data[30]},{data[31]}", file=sample)



    
    # if ((df.iloc[j]['Age']-writer_var[1])/writer_var[1]*100.0 < (100.0 - data_similarity) and (df.iloc[j]['Duration']-writer_var[4])/writer_var[4]*100.0 < (100.0 - data_similarity) and (df.iloc[j]['MotionSickness']-writer_var[5])/writer_var[5]*100.0 < (100.0 - data_similarity) and (df.iloc[j]['ImmersionLevel']-writer_var[6])/writer_var[6]*100.0 < (100.0 - data_similarity)):

    similarity = Differentiator(df, data[1], data[8], data[3], data[7], data[30], data[26], data[27] , data[28], data[29], data_similarity)
        
    if (similarity == 1):

        print(f"{index},{data[1]},{data[2]},{data[3]},{data[4]},{data[5]},{data[6]},{data[7]},{data[8]},{data[9]},{data[10]},{data[11]},{data[12]},{data[13]},{data[14]},{data[15]},{data[16]},{data[17]},{data[18]},{data[19]},{data[20]},{data[21]},{data[22]},{data[23]},{data[24]},{data[25]},{data[26]},{data[27]},{data[28]},{data[29]},{data[30]},{data[31]}", file=sample_realistic)
        index = index + 1




