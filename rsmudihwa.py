#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd 
import numpy as np
df=pd.read_csv('players_20.csv')


# In[36]:


df


# In[104]:


##check for missing data
df.isnull().sum()


# In[106]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
encoder = LabelEncoder()
categorical_data=df['preferred_foot']
categorical_encoded=encoder.fit_transform(categorical_data)
print(categorical_encoded)
###
encoderr = OneHotEncoder()
categorical_data_hot= encoderr.fit_transform(categorical_encoded.reshape(-1,1))
categorical_data_hot.toarray()


# In[107]:


encoder = LabelEncoder()
work_rate_en= df['work_rate']
categorical_encoded_wr=encoder.fit_transform(work_rate_en)
print(categorical_encoded_wr)
###
encoderrr = OneHotEncoder()
categorical_data_hot_wr= encoderrr.fit_transform(categorical_encoded_wr.reshape(-1,1))
categorical_data_hot_wr.toarray()


# In[5]:


#player stat features
columns = ['overall','attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure', 'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']
FIFA_data = df[columns]
FIFA_data.head()


# In[38]:


correlation=FIFA_data.corr()['overall'].sort_values(ascending=False)

##absolute value
abs_corr=abs(correlation)

###selecting relavant feature
relavant_features = pd.DataFrame(abs_corr[abs_corr>0.05])
relavant_features


# In[77]:


#taking the important features and other relevant features
columns = ["overall","movement_reactions","mentality_composure","power_shot_power","mentality_vision","attacking_short_passing","skill_ball_control","power_long_shots","skill_curve","value_eur","age","movement_acceleration","potential","skill_long_passing"]
FIFA_data = df[columns]
FIFA_data.head()


# In[78]:


X = FIFA_data.drop('overall',axis=1)
Y = FIFA_data['overall']


# In[79]:


X.head()


# In[80]:


Y.head()


# In[81]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scale = MinMaxScaler()


# In[82]:


scal = MinMaxScaler()
scaled = scal.fit_transform(X,Y)
scaled


# In[83]:


corr_matrix = FIFA_data.corr()
corr_matrix


# In[84]:


correlation = corr_matrix["overall"].sort_values(ascending=False)


# In[85]:


FIFA_data.info()


# In[86]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(FIFA_data, test_size=0.2, random_state=42)
print(len(train_set), "train +", len(test_set), "test")


# In[87]:


Xtrain,Xtest,Ytrain,Ytest = train_test_split(scaled,Y,test_size=0.2, random_state=42)


# In[88]:


Xtrain.shape


# In[89]:


Ytrain.shape


# In[90]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(Xtrain,Ytrain)


# In[91]:


lm = LinearRegression()
model = lm.fit(X,Y)
predictions = lm.predict(X)


# In[92]:


model =  LinearRegression()
model.fit(Xtrain,Ytrain)


# In[93]:


y_predict = model.predict(Xtest)
y_predict


# In[94]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(Ytest, y_predict))
print('MSE:', metrics.mean_squared_error(Ytest, y_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Ytest, y_predict)))


# In[95]:


#validating model for linear regression
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg, Xtrain, Ytrain,scoring='neg_mean_squared_error', cv=10)
lin_reg_scores = np.sqrt(-scores)
def display_scores(scores):
  print("Scores:", scores)
  print("Mean:", scores.mean())
  print("Standard Deviation:", scores.std())
display_scores(lin_reg_scores)


# In[96]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(Xtrain, Ytrain)


# In[97]:


from sklearn.metrics import mean_squared_error
FIFA_predictions = tree_reg.predict(Xtrain)
tree_mse = mean_squared_error(Ytrain, FIFA_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[98]:


#validating model for decision tree
scores = cross_val_score(tree_reg, Xtrain, Ytrain,
scoring='neg_mean_squared_error', cv=10)
tree_scores = np.sqrt(-scores)
display_scores(tree_scores)


# In[99]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(Xtrain, Ytrain)


# In[100]:


FIFA_predictions = forest_reg.predict(Xtrain)
forest_mse = mean_squared_error(Ytrain, FIFA_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[101]:


##validating model for random forest
scores = cross_val_score(forest_reg, Xtrain, Ytrain,
scoring='neg_mean_squared_error', cv=10)
forest_scores = np.sqrt(-scores)
display_scores(forest_scores)


# In[102]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error


# In[108]:


#Training using xgboost
data_dmatrix = xgb.DMatrix(data=X,label=Y)


# In[109]:


xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)


# In[110]:


xg_reg.fit(Xtrain,Ytrain)
preds = xg_reg.predict(Xtest)


# In[111]:


rmse = np.sqrt(mean_squared_error(Ytest, preds))
print("RMSE: %f" % (rmse))


# In[112]:


#validating xgbost
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)


# In[113]:


cv_results.head()


# In[114]:


import joblib
joblib.dump(forest_reg,'model.pkl')


# In[116]:


#testing data 
FIFA_15 = pd.read_csv("players_15.csv")
FIFA_15.head()


# In[117]:


FIFA_15.fillna(0,inplace=True)
FIFA_15.head()


# In[121]:


columns = ["overall","movement_reactions","mentality_composure","power_shot_power","mentality_vision","attacking_short_passing","skill_ball_control","power_long_shots","skill_curve","value_eur","age","movement_acceleration","potential","skill_long_passing"]
FIFA_15 = df[columns]
FIFA_15.head()


# In[122]:


x = FIFA_15.drop('overall',axis=1)
y = FIFA_15['overall']


# In[123]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scale = MinMaxScaler()


# In[124]:


scal = MinMaxScaler()
scaled = scal.fit_transform(X,Y)
scaled


# In[125]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(FIFA_15, test_size=0.2, random_state=42)
print(len(train_set), "train +", len(test_set), "test")


# In[126]:


xtrain,xtest,ytrain,ytest = train_test_split(scaled,y,test_size=0.2, random_state=42)


# In[127]:


# Use the forest's predict method on the test data
predictions =forest_reg .predict(xtest)


# In[128]:


# Calculate the absolute errors
errors = abs(predictions - ytest)


# In[129]:


# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[130]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / ytest)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:





# In[ ]:




