#project: p7
#submitter: bpilon
#partner: None
#hours:12

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import make_column_transformer

class UserPredictor:
    def __init__(self):
        self.cls = LogisticRegression(max_iter = 4000)
        self.x_cols = ['age', 'past_purchase_amt', '/backpack.html',
           '/basketball.html', '/bicycle.html', '/blender.html', '/chair.html',
           '/cleats.html', '/cooler.html', '/desk.html', '/keyboard.html',
           '/lamp.html', '/laptop.html', '/microwave.html', '/monitor.html',
           '/mug.html', '/printer.html', '/racquet.html', '/sneakers.html',
           '/spatula.html', '/spoon.html', '/tablet.html']
    def fit(self, users, logs, clicks):
        self.users = users
        self.logs = logs
        self.clicks = clicks
        users_res = pd.merge(self.users, self.clicks, on = 'id', how = 'right')
        web_info = self.logs[['id', 'url_visited', 'minutes_on_page']]
        url_info = pd.pivot_table(web_info,values='minutes_on_page',index='id',columns='url_visited')
        user_time_info = pd.merge(users_res,url_info, on = 'id', how = 'left').fillna(0)
        self.cls.fit(user_time_info[self.x_cols],user_time_info["y"])

        
    def predict(self, test_users, test_logs):
        self.test_users = test_users
        self.test_logs = test_logs
        web_info_test = self.test_logs[['id', 'url_visited', 'minutes_on_page']]
        url_info_test = pd.pivot_table(web_info_test,values='minutes_on_page',index='id',columns='url_visited')
        user_time_info_test = pd.merge(self.test_users,url_info_test, on = 'id', how = 'left').fillna(0)
        user_time_info_test['y'] = self.cls.predict(user_time_info_test[self.x_cols])
        array = user_time_info_test['y'].values
        return array
        
        
