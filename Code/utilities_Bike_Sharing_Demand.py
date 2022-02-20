import pandas as pd
import numpy as np

def get_home_dir():
    return "/home/vss/Documents/Personal/Personal Learnings - Data Science/OnlineCompetitions/Bike Sharing Demand/"

def get_data(home_dir):
    data_dir = f"{home_dir}Input/"
    train = pd.read_csv(f"{data_dir}train.csv")
    test = pd.read_csv(f"{data_dir}test.csv")
    sampleSubm = pd.read_csv(f"{data_dir}sampleSubmission.csv")
    return train, test, sampleSubm

def divide_train_validation(validation_month_days = [8,  9, 10, 11, 12, 13]):
    mask = train['which_month_day?'].isin(validation_month_days)
    return train[~mask].reset_index(drop=True), train[mask].reset_index(drop=True)

def features_extraction(data):
    data['which_month_day?'] = data['datetime'].dt.day
    data['which_hour?'] = data['datetime'].dt.hour
    data['which_week_day?'] = data['datetime'].dt.weekday
    data['which_year?'] = data['datetime'].dt.year
    data['which_month?'] = data['datetime'].dt.month
    return data

def get_feature_importance_random_forest(features, feature_importances_):
    feature_imp = pd.DataFrame(
        {'features': features,
         'importance': feature_importances_}
    ).sort_values(['importance'],ascending=[False])
    feature_imp['imp_cum_sum'] = feature_imp['importance'].cumsum()
    print(feature_imp)


def fit_predict(model, train_set, test_set, features, target):
    model.fit(train_set[features], np.log(train_set[target[0]] + 1))
    pred_casual_train = np.exp(model.predict(train_set[features])) - 1
    pred_casual = np.exp(model.predict(test_set[features])) - 1

    model.fit(train_set[features], np.log(train_set[target[1]] + 1))
    pred_reg_train = np.exp(model.predict(train_set[features])) - 1
    pred_reg = np.exp(model.predict(test_set[features])) - 1

    count_train = pred_casual_train + pred_reg_train
    count_test = pred_casual + pred_reg
    count_train[count_train < 0] = 0
    count_test[count_test < 0] = 0
    print("Train RMLSE", np.sqrt(mean_squared_log_error(train_set['count'], count_train)).round(2))
    #print(test_set.columns.isin(['count']).sum())
    #print(test_set.columns)
    if test_set.columns.isin(['count']).sum() != 0:
       print("Test RMLSE", np.sqrt(mean_squared_log_error(test_set['count'], count_test)).round(2))
    return count_test, count_train