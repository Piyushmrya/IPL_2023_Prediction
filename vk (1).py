#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# loading all datasets
df1 = pd.read_csv('all_bowlers.csv')
df2 = pd.read_csv('all_batters.csv')
df3 = pd.read_csv('ipl_2023_ball.csv')
df4 = pd.read_csv('ipl_2023_bat.csv')


# In[2]:


# # adding a feature to incorporate if particular opponent is playing that year or not
# arr_bowlers = []
# for bowler in df1.bowler:
#     if bowler in list(df3.player):
#         arr_bowlers.append(1)
#     else:
#         arr_bowlers.append(0)
        
# arr_batters = []
# for batter in df2.batter:
#     if batter in list(df4.player):
#         arr_batters.append(1)
#     else:
#         arr_batters.append(0)

# df1['bowler_2023'] = arr_bowlers
# df2['batter_2023'] = arr_batters  

# ##

# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()
# le = le.fit(df1['bowler'])
# df1['bowler_le'] = le.transform(df1['bowler'])

# le2 = le.fit(df2['batter'])
# df2['batter_le'] = le.transform(df2['batter'])

grouped1 = df1.groupby(df1.batter)
grouped2 = df2.groupby(df2.bowler)


# In[ ]:





# In[3]:


df2


# In[4]:


from sklearn.ensemble import RandomForestClassifier

batting = ['RD Gaikwad', 'DP Conway', 'S Dube', 'SN Khan', 'AM Rahane', 'BA Stokes']
bowling = ['YS Chahal', 'KK Ahmed', 'A Nortje', 'KK Ahmed', 'KK Ahmed', 'TU Deshpande']

counts_batter = df1.batter.value_counts()
counts_bowler = df2.bowler.value_counts()

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
print(runs)            


# In[ ]:


df2[df2['batter']==batting[bat_pl]].reset_index(drop=True)['batter_2023']


# In[55]:


counts_bowler['TU Deshpande']


# In[ ]:





# In[7]:


# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import make_scorer, accuracy_score

# df5 = pd.DataFrame()
# df6 = pd.DataFrame()
# counts_batter = df1.batter.value_counts()
# counts_bowler = df2.bowler.value_counts()

# batsman = []
# accuracy = []
# pred_runs_sum = []
# actual_runs = []
# balls = []
# for batter in list(df4.player):
#     if batter in list(df2.batter) and counts2[batter] > 5:
#         df_g = grouped2.get_group(batter)
#         x = df_g.drop(['team','batter','bowler','isWicketDelivery','batsman_run'], axis=1)
#         y = df_g['batsman_run']
#         x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
#         clf = RandomForestClassifier()
#         clf.fit(x_train,y_train)
#         p = clf.predict(x_test)
#         batsman.append(batter)
#         accuracy.append(accuracy_score(abs(p), y_test))
#         p_sum.append(sum(p))
#         y_t.append(sum(y_test))
#         balls.append(len(y_test))

# d2['batter'] = batsman
# d2['accuracy'] = accuracy
# d2['p_sum'] = p_sum
# d2['y_test_sum'] = y_t
# d2['balls'] = balls

# bowler_a = []
# accuracy = []
# p_sum = []
# y_t = []
# balls = []
# for bowler in list(df3.player):
#     if bowler in list(df5.bowler) and counts5[bowler] > 5:
#         df_g = grouped5.get_group(bowler)
#         x = df_g.drop(['team','batter','bowler','isWicketDelivery'], axis=1)
#         y = df_g['isWicketDelivery']
#         x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
#         clf = RandomForestClassifier()
#         clf.fit(x_train,y_train)
#         p = clf.predict(x_test)
#         bowler_a.append(bowler)
#         accuracy.append(accuracy_score(abs(p), y_test))
#         p_sum.append(abs(sum(p)))
#         y_t.append(sum(y_test))
#         balls.append(len(y_test))

# d5['bowler'] = bowler_a
# d5['accuracy'] = accuracy
# d5['p_sum'] = p_sum
# d5['y_test_sum'] = y_t
# d5['balls'] = balls

