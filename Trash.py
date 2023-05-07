import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.write("""
# IPL Run prediction WebApp
""")

# Data loading

df2 = pd.read_csv('ipl_files/all_bowlers.csv')
df3 = pd.read_csv('ipl_files/ipl_2023_ball.csv')
df4 = pd.read_csv('ipl_files/ipl_2023_bat.csv')
df5 = pd.read_csv('ipl_files/all_batters.csv')

# batting = list(batting)
# bowling = list(bowling)
arr_bowlers = []
for bowler in df2.bowler:
    if bowler in list(df3.player):
        arr_bowlers.append(1)
    else:
        arr_bowlers.append(0)
        
arr_batters = []
for batter in df5.batter:
    if batter in list(df4.player):
        arr_batters.append(1)
    else:
        arr_batters.append(0)

# Adding it to the dataframe        

df2['bowler_2023'] = arr_bowlers
df5['batter_2023'] = arr_batters

#Encoding Bowlers and batters names

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le = le.fit(df2['bowler'])
df2['bowler_le'] = le.transform(df2['bowler'])

le2 = le.fit(df5['batter'])
df5['batter_le'] = le.transform(df5['batter'])

#Group them 

grouped2 = df2.groupby(df2.batter)
grouped5 = df5.groupby(df5.bowler)

# Random 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
d2 = pd.DataFrame()
d5 = pd.DataFrame()
counts2 = df2.batter.value_counts()
counts5 = df5.bowler.value_counts()

# Getting user input to select players 

batting = st.multiselect("Select bowler names" , df3.player,max_selections=6)
bowling = st.multiselect("Select batter names" , df4.player,max_selections=6)
st.write(batting)
st.write(bowling)