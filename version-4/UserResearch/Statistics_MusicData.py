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

df = pd.read_csv("UserResearch\MusicExperienceData.csv")

print(df.head())

df_reduced = df['Age'].copy()

print(df_reduced)

print("Age statistics:")
print("Count: ", df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: " , df_reduced.std())


print("")
df_reduced = df['Hours per day'].copy()


print("Hours per day statistics:")
print("Count Duration: " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())

print("")
df_reduced = df['BPM'].copy()


print("BPM statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())

print("")
df_reduced = df[df['OCD'] != 0].copy()
df_reduced = df_reduced['OCD']


print("OCD statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())
print("Maximum: ", df_reduced.max())
print("Minimum: ", df_reduced.min())

print("")

df_reduced = df[df['Anxiety'] != 0].copy()
df_reduced = df_reduced['Anxiety']


print("Anxiety statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())
print("Maximum: ", df_reduced.max())
print("Minimum: ", df_reduced.min())


print("")

df_reduced = df[df['Depression'] != 0].copy()
df_reduced = df_reduced['Depression']


print("Depression statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())
print("Maximum: ", df_reduced.max())
print("Minimum: ", df_reduced.min())

print("")
df_reduced = df[df['Insomnia'] != 0].copy()
df_reduced = df_reduced['Insomnia']


print("Insomnia statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())
print("Maximum: ", df_reduced.max())
print("Minimum: ", df_reduced.min())


df_reduced = df[(df['OCD'] != 0) &  (df['Anxiety'] == 0) & (df['Depression'] == 0) & (df['Insomnia'] == 0)].copy()
df_reduced = df_reduced[["Fav genre", "Music effects"]].copy()
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print("Number of healthy cases: ", df_reduced.count())
#print(df_reduced)

df_reduced = df[(df['OCD'] == 0) &  (df['Anxiety'] != 0) & (df['Depression'] == 0) & (df['Insomnia'] == 0)].copy()
df_reduced = df_reduced[["Fav genre", "Music effects"]].copy()
# df_reduced = df_reduced[["Fav genre","Frequency [Classical]","Frequency [Country]","Frequency [EDM]","Frequency [Folk]","Frequency [Gospel]","Frequency [Hip hop]","Frequency [Jazz]","Frequency [K pop]","Frequency [Latin]","Frequency [Lofi]","Frequency [Metal]","Frequency [Pop]","Frequency [R&B]","Frequency [Rap]","Frequency [Rock]","Frequency [Video game music]"]].copy()
#df_reduced = df_reduced[["Frequency [Classical]","Frequency [Country]","Frequency [EDM]"]].copy()
#reduced_df = df_reduced.replace({'Sometimes': 2.0, 'Rarely': 1.0, 'Never': 0.0, 'Very frequently': 3.0, 'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print(df_reduced)

df_reduced = df[(df['OCD'] == 0) &  (df['Anxiety'] == 0) & (df['Depression'] != 0) & (df['Insomnia'] == 0)].copy()
df_reduced = df_reduced[["Fav genre", "Music effects"]].copy()
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print("Number of healthy cases: ", df_reduced.count())
#print(df_reduced)

df_reduced = df[(df['OCD'] == 0) &  (df['Anxiety'] == 0) & (df['Depression'] == 0) & (df['Insomnia'] != 0)].copy()
df_reduced = df_reduced[["Fav genre", "Music effects"]].copy()
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print("Number of healthy cases: ", df_reduced.count())
#print(df_reduced)



df_reduced = df[(df['OCD'] != 0)].copy()
df_reduced = df_reduced[["Fav genre", "Music effects"]].copy()
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print("Number of healthy cases: ", df_reduced.count())
#print(df_reduced)

df_reduced = df[(df['Anxiety'] != 0)].copy()
df_reduced = df_reduced[["Fav genre", "Music effects"]].copy()
# df_reduced = df_reduced[["Fav genre","Frequency [Classical]","Frequency [Country]","Frequency [EDM]","Frequency [Folk]","Frequency [Gospel]","Frequency [Hip hop]","Frequency [Jazz]","Frequency [K pop]","Frequency [Latin]","Frequency [Lofi]","Frequency [Metal]","Frequency [Pop]","Frequency [R&B]","Frequency [Rap]","Frequency [Rock]","Frequency [Video game music]"]].copy()
#df_reduced = df_reduced[["Frequency [Classical]","Frequency [Country]","Frequency [EDM]"]].copy()
#reduced_df = df_reduced.replace({'Sometimes': 2.0, 'Rarely': 1.0, 'Never': 0.0, 'Very frequently': 3.0, 'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print(df_reduced)

df_reduced = df[(df['Depression'] != 0)].copy()
df_reduced = df_reduced[["Fav genre", "Music effects"]].copy()
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print("Number of healthy cases: ", df_reduced.count())
#print(df_reduced)

df_reduced = df[(df['Insomnia'] != 0)].copy()
df_reduced = df_reduced[["Fav genre", "Music effects"]].copy()
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print("Number of healthy cases: ", df_reduced.count())
#print(df_reduced)


df_reduced = df[(df['OCD'] != 0) &  (df['Anxiety'] == 0) & (df['Depression'] == 0) & (df['Insomnia'] == 0)].copy()
df_reduced = df_reduced[["Hours per day", "Music effects"]].copy()
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print("Number of healthy cases: ", df_reduced.count())
#print(df_reduced)

df_reduced = df[(df['OCD'] == 0) &  (df['Anxiety'] != 0) & (df['Depression'] == 0) & (df['Insomnia'] == 0)].copy()
df_reduced = df_reduced[["Hours per day", "Music effects"]].copy()
# df_reduced = df_reduced[["Fav genre","Frequency [Classical]","Frequency [Country]","Frequency [EDM]","Frequency [Folk]","Frequency [Gospel]","Frequency [Hip hop]","Frequency [Jazz]","Frequency [K pop]","Frequency [Latin]","Frequency [Lofi]","Frequency [Metal]","Frequency [Pop]","Frequency [R&B]","Frequency [Rap]","Frequency [Rock]","Frequency [Video game music]"]].copy()
#df_reduced = df_reduced[["Frequency [Classical]","Frequency [Country]","Frequency [EDM]"]].copy()
#reduced_df = df_reduced.replace({'Sometimes': 2.0, 'Rarely': 1.0, 'Never': 0.0, 'Very frequently': 3.0, 'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print(df_reduced)

df_reduced = df[(df['OCD'] == 0) &  (df['Anxiety'] == 0) & (df['Depression'] != 0) & (df['Insomnia'] == 0)].copy()
df_reduced = df_reduced[["Hours per day", "Music effects"]].copy()
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print("Number of healthy cases: ", df_reduced.count())
#print(df_reduced)

df_reduced = df[(df['OCD'] == 0) &  (df['Anxiety'] == 0) & (df['Depression'] == 0) & (df['Insomnia'] != 0)].copy()
df_reduced = df_reduced[["Hours per day", "Music effects"]].copy()
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print("Number of healthy cases: ", df_reduced.count())
#print(df_reduced)


df_reduced = df[(df['OCD'] != 0) &  (df['Anxiety'] <= 2) & (df['Depression'] <= 2) & (df['Insomnia'] <= 2)].copy()
df_reduced = df_reduced[["Age", "Music effects"]].copy()
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print("Number of healthy cases: ", df_reduced.count())
#print(df_reduced)

df_reduced = df[(df['OCD'] == 0) &  (df['Anxiety'] != 0) & (df['Depression'] == 0) & (df['Insomnia'] == 0)].copy()
df_reduced = df_reduced[["Age", "Music effects"]].copy()
# df_reduced = df_reduced[["Fav genre","Frequency [Classical]","Frequency [Country]","Frequency [EDM]","Frequency [Folk]","Frequency [Gospel]","Frequency [Hip hop]","Frequency [Jazz]","Frequency [K pop]","Frequency [Latin]","Frequency [Lofi]","Frequency [Metal]","Frequency [Pop]","Frequency [R&B]","Frequency [Rap]","Frequency [Rock]","Frequency [Video game music]"]].copy()
#df_reduced = df_reduced[["Frequency [Classical]","Frequency [Country]","Frequency [EDM]"]].copy()
#reduced_df = df_reduced.replace({'Sometimes': 2.0, 'Rarely': 1.0, 'Never': 0.0, 'Very frequently': 3.0, 'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print(df_reduced)

df_reduced = df[(df['OCD'] <= 2) &  (df['Anxiety'] <= 2) & (df['Depression'] != 0) & (df['Insomnia'] <= 2 )].copy()
df_reduced = df_reduced.dropna()
df_reduced = df_reduced[["Age", "Music effects"]].copy()
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0,' ':0.0,'':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
#print("Number of healthy cases: ", df_reduced.count())
#print(df_reduced)

df_reduced = df[(df['OCD'] <= 2) &  (df['Anxiety'] <= 2) & (df['Depression'] <= 2 ) & (df['Insomnia'] != 0)].copy()
df_reduced = df_reduced[["Age", "Music effects"]].copy()
reduced_df = df_reduced.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0})
print(reduced_df.corr("pearson"))
print(df_reduced.count())

#print("Number of healthy cases: ", df_reduced.count())
#print(df_reduced)

#df_reduced = df['Age'].copy()
#stats_df = []

#for index in df_reduced.iterrows():
#    stats_df.append(df_reduced.iloc[index, 1])
        
#plt.hist(df_reduced,bins=100)
#plt.show()

#df_reduced = df['Hours per day'].copy()

#for index in df_reduced.iterrows():
#    stats_df.append(df_reduced.iloc[index, 1])
        
#plt.hist(df_reduced,bins=100)
#plt.show()

#df_reduced = df['Music effects'].copy()
#df_reduced = df[(df['Fav genre'] == 'Video game music')].copy()
#df_bpm = df_reduced['BPM'].copy()
#print(df_bpm.mean())
#print(df_bpm.std())
#df_reduced = df_reduced.dropna()

#for index in df_reduced.iterrows():
#    stats_df.append(df_reduced.iloc[index, 1])
#print(df_reduced.max())
#print(df_reduced.min())
#plt.hist(df_reduced,bins=5)
#plt.show()


#df_stats = df[['Music effects', 'Composer', 'Exploratory', 'Foreign languages', 'Instrumentalist', 'Fav genre', 'Frequency [Classical]']].copy()
#df_stats = df_stats.replace({'Improve':3.0, 'No effect': 2.0, 'Worsen':0.0, 'NaN': 0.0,'Rock': 1.0, 'Country': 10.0, 'Classical': 10.0,'EDM' : 10.0, 'Folk' : 10.0,'Gospel' : 10.0,'Hip hop' : 10.0,'Jazz' : 10.0,'K pop' : 10.0,'Latin' : 10.0,'Lofi' : 10.0,'Metal' : 10.0,'Pop' : 10.0,'R&B' : 10.0,'Rap' : 10.0,'Video game music' : 10.0, 'Yes': 1.0, 'No': 0.0, 'Never': 0.0, 'Sometimes': 2.0, 'Rarely':1.0, 'Very frequently': 4.0})
#print(df_stats.corr('pearson'))
#print(df_reduced.value_counts())


df = pd.read_csv("UserResearch\MusicModelData.csv")

print(df.head())

df_reduced = df['Age'].copy()

print(df_reduced)

print("Age statistics:")
print("Count: ", df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: " , df_reduced.std())


print("")
df_reduced = df['Hours per day'].copy()


print("Hours per day statistics:")
print("Count Duration: " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())

print("")
df_reduced = df['BPM'].copy()


print("BPM statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())

print("")
df_reduced = df[df['OCD'] != 0].copy()
df_reduced = df_reduced['OCD']


print("OCD statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())
print("Maximum: ", df_reduced.max())
print("Minimum: ", df_reduced.min())

print("")

df_reduced = df[df['Anxiety'] != 0].copy()
df_reduced = df_reduced['Anxiety']


print("Anxiety statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())
print("Maximum: ", df_reduced.max())
print("Minimum: ", df_reduced.min())


print("")

df_reduced = df[df['Depression'] != 0].copy()
df_reduced = df_reduced['Depression']


print("Depression statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())
print("Maximum: ", df_reduced.max())
print("Minimum: ", df_reduced.min())

print("")
df_reduced = df[df['Insomnia'] != 0].copy()
df_reduced = df_reduced['Insomnia']


print("Insomnia statistics:")
print("Count : " , df_reduced.count())
print("Mean: ", df_reduced.mean())
print("Standard deviation: ", df_reduced.std())
print("Maximum: ", df_reduced.max())
print("Minimum: ", df_reduced.min())
