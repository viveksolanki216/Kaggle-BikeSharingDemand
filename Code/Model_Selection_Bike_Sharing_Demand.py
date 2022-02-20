import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# ---------------- Baseline Model --------------------------------
### Random Forest Regressor Gives a baseline with 0.4 RMSLE, Details
# Untuned, with default parameters

# Features and their importance
# ------------------------------------
#  features  importance  imp_cum_sum
#        which_hour?    0.575502     0.575502
#               temp    0.182491     0.757993
#           humidity    0.050460     0.808452
#         workingday    0.044314     0.852767
#              atemp    0.040859     0.893625
#   which_week_day?    0.028562     0.922187
#          windspeed    0.019571     0.941758
#   which_month_day?    0.018724     0.960482
#             season    0.014902     0.975384
#       which_year?    0.012003     0.987387
#            weather    0.011965     0.999352
#            holiday    0.000648     1.000000

# Steps :
# Divide train set into train and validation set, not randomly, but as the train set and test set is in 19:11 ratio
# Divide train, validation into 14:6, (in between)

home_dir = get_home_dir()
train, test, sample_sub = get_data(home_dir)
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])
train = features_extraction(train)
test = features_extraction(test)

features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',
            'which_month_day?', 'which_hour?', 'which_week_day?', 'which_year?', 'which_month?']
target = ['casual', 'registered']

print(train['which_month_day?'].unique())
validation_list =  [[1,  2,  3,  4,  5,  6],  [7,  8,  9, 10, 11, 12], [13, 14, 15, 16, 17, 18, 19]]

#poly_reg = Pipeline([
#    ("PF", PolynomialFeatures(degree=2, include_bias=False)),
#    ("Scaler", StandardScaler()),
#    ("Linear Regression", LinearRegression()),
#])

tree_models = [
    #["RF", RandomForestRegressor(n_jobs=5, n_estimators=100,
    #                             min_samples_leaf=10)],
    #["Gradient Boosting", AdaBoostRegressor()],
    ["Gradient Boosting", GradientBoostingRegressor(n_estimators=300,
        criterion='mse', min_samples_leaf=30, max_depth=3)]
]

# Cross validation
for modelname, model in tree_models:
    print(modelname)
    for validation_month_days in validation_list:
        print(validation_month_days)
        train_set, valid_set = divide_train_validation(validation_month_days)
        valid_set['pred_count'], train_set['pred_count'] = fit_predict(
            model, train_set, valid_set, features, target
        )

test['count_pred'], train['count_pred'] = fit_predict( model, train, test, features, target)
test.rename(columns={'count_pred':'count'})[['datetime', 'count']].to_csv(f'{home_dir}/Output/GB_200_mse_min_leaf20_depth4.csv', index=False)

from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',
            'which_month_day?', 'which_hour?', 'which_week_day?', 'which_year?', 'which_month?']
num_features = ['temp', 'atemp', 'humidity', 'windspeed']
cat_features = ['season', 'holiday', 'workingday', 'weather', 'which_month_day?', 'which_hour?',
                'which_week_day?', 'which_year?', 'which_month?']
target = ['casual', 'registered']

encode = OneHotEncoder()
cat = encode.fit_transform(train[cat_features]).toarray()
num = train[num_features].values
X = np.concatenate([num, cat], axis=1)

pca = make_pipeline(PCA(n_components=1), LinearRegression())

pca.fit(X, train['count'])
train['count_pred'] = pca.predict(X)
train.loc[train['count_pred'] <0, 'count_pred' ]=0
np.sqrt(mean_squared_log_error(train['count'], train['count_pred']))

pca['truncatedsvd'].explained_variance_ratio_


# Random Forest seems to have best (equivalent to gradient boosting). Lets optimize the performance for
    # the parameters using the GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
rmsle = make_scorer(mean_squared_log_error)
rf = tree_models[0][1]
parameters = {'n_estimators':[50,100,250,500], 'min_samples_leaf':[30, 50, 100]}
grid_m = GridSearchCV(rf, parameters, scoring=rmsle)
grid_m.fit(train[features], np.log(train[target[0]] + 1))
temp0 = pd.DataFrame(grid_m.cv_results_)

grid_m.fit(train[features], np.log(train[target[1]] + 1))
temp1 = pd.DataFrame(grid_m.cv_results_)


["Linear Regression", make_pipeline(StandardScaler(), LinearRegression())],

    ["Poly Regression", make_pipeline(
        PolynomialFeatures(degree=3, include_bias=False),
        StandardScaler(),
        TruncatedSVD(n_components=10),
        LinearRegression())]

    train_set = train
    rf0 = RandomForestRegressor()
    rf0.fit(train_set[features], train_set[target[0]])
    pred_casual_train = rf0.predict(train_set[features])
    pred_casual = rf0.predict(test[features])

    rf1 = RandomForestRegressor()
    rf1.fit(train_set[features], train_set[target[1]])
    pred_reg_train = rf1.predict(train_set[features])
    pred_reg = rf1.predict(test[features])

    train_set['count_pred'] = pred_casual_train + pred_reg_train
    test['count'] = pred_casual + pred_reg
    print("RMLSE", np.sqrt(mean_squared_log_error(train_set['count'], train_set['count_pred'])).round(2))

    test[['datetime', 'count']].to_csv(f'{home_dir}/Output/RF_Baseline.csv', index=False)