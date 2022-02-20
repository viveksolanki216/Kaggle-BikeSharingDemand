import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error


# Steps :
# Divide train set into train and validation set, not randomly, but as the train set and test set is in 19:11 ratio
# Divide train, validation into 14:6, (in between)
#
home_dir = get_home_dir()
train, test, _ = get_data(home_dir)
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])
train = features_extraction(train)

train_set, valid_set = divide_train_validation(validation_month_days = [8,  9, 10, 11, 12, 13])

features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
target = ['casual', 'registered']

rf0 = RandomForestRegressor()
rf0.fit(train_set[features], train_set[target[0]])
pred_casual = rf0.predict(valid_set[features])

rf1 = RandomForestRegressor()
rf1.fit(train_set[features], train_set[target[1]])
pred_reg = rf1.predict(valid_set[features])

print("RMLSE", np.sqrt(mean_squared_log_error(valid_set['count'], pred_casual+pred_reg)))

get_feature_importance_random_forest(features, rf0.feature_importances_)
get_feature_importance_random_forest(features, rf1.feature_importances_)


features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',
            'which_month_day?', 'which_hour?', 'which_week_day?', 'which_year?']
target = ['casual', 'registered']

rf0 = RandomForestRegressor()
rf0.fit(train_set[features], train_set[target[0]])
pred_casual = rf0.predict(valid_set[features])

rf1 = RandomForestRegressor()
rf1.fit(train_set[features], train_set[target[1]])
pred_reg = rf1.predict(valid_set[features])

print("RMLSE", np.sqrt(mean_squared_log_error(valid_set['count'], pred_casual+pred_reg)))

get_feature_importance_random_forest(features, rf0.feature_importances_)
get_feature_importance_random_forest(features, rf1.feature_importances_)



features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',
            'which_month_day?', 'which_hour?', 'which_week_day?', 'which_year?']
target = ['casual', 'registered']

rf0 = RandomForestRegressor()
rf0.fit(train_set[features], np.log(train_set[target[0]]+1))
pred_casual = rf0.predict(valid_set[features])

rf1 = RandomForestRegressor()
rf1.fit(train_set[features], np.log(train_set[target[1]]+1))
pred_reg = rf1.predict(valid_set[features])

print("RMLSE", np.sqrt(mean_squared_log_error(valid_set['count'], np.exp(pred_casual)-1+np.exp(pred_reg)-1)))

get_feature_importance_random_forest(features, rf0.feature_importances_)
get_feature_importance_random_forest(features, rf1.feature_importances_)












