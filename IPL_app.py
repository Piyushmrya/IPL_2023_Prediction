import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.write("""
# IPL Run prediction WebApp
""")

# Data loading

df1 = pd.read_csv('all_bowlers.csv')
df2 = pd.read_csv('all_batters.csv')
df3 = pd.read_csv('ipl_2023_ball.csv')
df4 = pd.read_csv('ipl_2023_bat.csv')


# arr_bowlers = []
# for bowler in df2.bowler:
#     if bowler in list(df3.player):
#         arr_bowlers.append(1)
#     else:
#         arr_bowlers.append(0)
        
# arr_batters = []
# for batter in df5.batter:
#     if batter in list(df4.player):
#         arr_batters.append(1)
#     else:
#         arr_batters.append(0)

# # Adding it to the dataframe        

# df2['bowler_2023'] = arr_bowlers
# df5['batter_2023'] = arr_batters

# #Encoding Bowlers and batters names

# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()
# le = le.fit(df2['bowler'])
# df2['bowler_le'] = le.transform(df2['bowler'])

# le2 = le.fit(df5['batter'])
# df5['batter_le'] = le.transform(df5['batter'])

#Group them 

grouped1 = df1.groupby(df1.batter)
grouped2 = df2.groupby(df2.bowler)

# Random 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
counts_batter = df1.batter.value_counts()
counts_bowler = df2.bowler.value_counts()

# Getting user input to select players 

bowling = st.multiselect("Select bowler names" , df3.player,max_selections=6)
batting = st.multiselect("Select batter names" , df4.player,max_selections=6)

#The prediction funciton

def pred_fun():
    ballnumber = 0
    over = 0
    runs = 0
    innings = 2
    isWic = 0
    bat_pl = 0
    non_st = 1
    bowl_pl = 0
    run = 0
    for over in range(0,6):
        for ballnumber in range(1,7):
            
            if bowling[bowl_pl] in list(df2.bowler) and counts_bowler[bowling[bowl_pl]] > 5 and batting[bat_pl] in list(df2.batter):
                df_g = grouped2.get_group(bowling[bowl_pl])
                x = df_g.drop(['batter','team','bowler','isWicketDelivery'], axis=1)
                y = df_g['isWicketDelivery']
                clf = RandomForestClassifier()
                clf.fit(x, y)
                isWic = clf.predict([[innings, over, ballnumber, df2[df2['batter']==batting[bat_pl]].reset_index(drop=True)['batter_avg_wic_per_ball'][0] ]])

            if isWic == 1:
                bat_pl = max(bat_pl, non_st) + 1
                print('wic')
                isWic = 0
            else:
                if batting[bat_pl] in list(df1.batter) and counts_batter[batting[bat_pl]] > 5 and bowling[bowl_pl] in list(df1.bowler):
                    df_g = grouped1.get_group(batting[bat_pl])
                    x = df_g.drop(['team','batter','bowler','isWicketDelivery','batsman_run'], axis=1)
                    y = df_g['batsman_run']
                    clf2 = RandomForestClassifier()
                    clf2.fit(x,y)
                    run = clf2.predict([[ innings, over, ballnumber, df1[df1['bowler']==bowling[bowl_pl]].reset_index(drop=True)['avg_runs_per_ball'][0] ]])                     
                    print(run, batting[bat_pl], ballnumber)
                    runs += float(run)
                else:
                    runs += 10.0/6.0
            if run % 2 == 1:
                temp = bat_pl
                bat_pl = non_st
                non_st = temp
            if ballnumber == 6:
                temp = bat_pl
                bat_pl = non_st
                non_st = temp    
        bowl_pl += 1
    return runs


if len(batting)==6 and len(bowling)==6:
    # Prediction
    runs = pred_fun()
    #Output 
    st.subheader("Runs scored : ")
    st.write(runs)
else:
    st.write("Please select all the Batters and bowlers names")
