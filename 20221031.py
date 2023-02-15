import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal, Cauchy
from ngboost.scores import MLE, CRPScore

from numpy import concatenate

import joblib

df_train = pd.read_csv("./master_data.csv")

"""outlier"""
#df_test = pd.read_csv("./outlier_data.csv")

"""spike"""
#df_test = pd.read_csv("./spike_data.csv")

"""stuck-at [20.9~23.6] """
#df_test = pd.read_csv(./test_data.csv")
#df_test.AIR_TEMPERATURE[500:500+144] = 0

"""drift+"""
#df_test = pd.read_csv("./test_data.csv")
#drift = np.arange(0,5,0.0347)
#df_test.AIR_TEMPERATURE[500:500+144+1] = df_test.AIR_TEMPERATURE[500:500+144+1] + drift

"""drift-"""
#df_test = pd.read_csv("./test_data.csv")
#drift = np.arange(0,5,0.0347)
#df_test.AIR_TEMPERATURE[500:500+144+1] = df_test.AIR_TEMPERATURE[500:500+144+1] - drift

df_train[df_train['AIR_PRESSURE'] < 900] = np.nan
df_train[df_train['AIR_PRESSURE'] > 1100 ] = np.nan

df_train[df_train['AIR_TEMPERATURE'] < 1] = np.nan
df_train[df_train['AIR_TEMPERATURE'] > 70 ] = np.nan

df_train[df_train['HUMIDITY'] < 0] = np.nan
df_train[df_train['HUMIDITY'] > 100 ] = np.nan

df_train[df_train['WIND_SPEED'] < 0] = np.nan
df_train[df_train['WIND_SPEED'] > 80 ] = np.nan


#df_train['AIR_TEMPERATURE_delta'] = df_train['AIR_TEMPERATURE']-df_train['AIR_TEMPERATURE'].shift(1)
#df_train['AIR_PRESSURE_delta'] = df_train['AIR_PRESSURE']-df_train['AIR_PRESSURE'].shift(1)
#df_train['HUMIDITY_delta'] = df_train['HUMIDITY']-df_train['HUMIDITY'].shift(1)
#df_train['WIND_SPEED_delta'] = df_train['WIND_SPEED']-df_train['WIND_SPEED'].shift(1)
#
#df_train[df_train['AIR_TEMPERATURE_delta']>(df_train['AIR_TEMPERATURE_delta'].mean()*2)] = np.nan
#df_train[df_train['AIR_PRESSURE_delta']>(df_train['AIR_PRESSURE_delta'].mean()*2)] = np.nan
#df_train[df_train['HUMIDITY_delta']>(df_train['HUMIDITY_delta'].mean()*2)] = np.nan
#df_train[df_train['WIND_SPEED_delta']>(df_train['WIND_SPEED_delta'].mean()*2)] = np.nan

# 1. IQR
"""
q3 = df_test['AIR_TEMPERATURE'].quantile(0.75)
q1 = df_test['AIR_TEMPERATURE'].quantile(0.25)

iqr = q3-q1

def iqr_outlier(df):
    iqr_score = df['AIR_TEMPERATURE']
    if iqr_score > q3 + 1.5 * iqr or iqr_score < q1 - 1.5 * iqr:
        return True
    else:
        return False
df_test['AIR_TEMPERATURE_upper'] = q3 + 1.5 * iqr
df_test['AIR_TEMPERATURE_lower'] = q1 - 1.5 * iqr
df_test['IQR_이상치여부'] = df_test.apply(iqr_outlier, axis=1)

print(len(df_test[df_test['IQR_이상치여부'] == True]))

plt.figure(figsize=(22, 5))
x_axis = np.arange(len(df_test))
plt.ylim([5, 35])
plt.scatter(x_axis, df_test.AIR_TEMPERATURE, c='green',label='AIR_TEMPERATURE' ,lw=1)
plt.fill_between(x_axis, df_test.AIR_TEMPERATURE_lower,  df_test.AIR_TEMPERATURE_upper,label = 'IQR', color='gray', alpha=0.5)
plt.legend(fontsize=14)
plt.show()
"""


# 2. Smoothing 기법
"""
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

df_rolling = pd.DataFrame({'AIR_TEMPERATURE': df_test.AIR_TEMPERATURE, 'AIR_TEMPERATURE_20d': df_test.AIR_TEMPERATURE.rolling(window=20).mean()})
df_plus = pd.DataFrame({'AIR_TEMPERATURE_upper': df_rolling.AIR_TEMPERATURE_20d+df_rolling.AIR_TEMPERATURE.std()*1,
                        'AIR_TEMPERATURE_lower': df_rolling.AIR_TEMPERATURE_20d-df_rolling.AIR_TEMPERATURE.std()*1})
df_rolling = pd.concat([df_rolling,df_plus],axis=1)

df_rolling.dropna(axis=0, inplace=True)

print(len(df_rolling['AIR_TEMPERATURE']))
plt.figure(figsize=(22, 5))
x_axis = np.arange(len(df_rolling))
plt.ylim([-2, 35])
plt.scatter(x_axis, df_rolling.AIR_TEMPERATURE, c='green',label='AIR_TEMPERATURE' ,lw=1)
plt.plot(x_axis, df_rolling.AIR_TEMPERATURE_20d, 'b-', lw=2,label='AIR_TEMPERATURE_20d')
plt.fill_between(x_axis, df_rolling.AIR_TEMPERATURE_lower,  df_rolling.AIR_TEMPERATURE_upper,label = 'smoothing_1sigma', color='gray', alpha=0.5)
plt.legend(fontsize=14)
plt.show()

df_rolling.info()



#df_figure = df_rolling
#wrong = []
#real_wrong = []
#for i in range(1439):
#    if (df_figure.AIR_TEMPERATURE[i] >= df_figure.AIR_TEMPERATURE_upper[i]) or (df_figure.AIR_TEMPERATURE[i] <= df_figure.AIR_TEMPERATURE_lower[i]):
#     wrong.append(i)
#    if (df_figure.AIR_TEMPERATURE[i] < 20.9) or (df_figure.AIR_TEMPERATURE[i] > 23.6):
#     real_wrong.append(i)
#
#print(wrong)
#print(len(wrong))
#print(real_wrong)
#print(len(real_wrong))
"""
# 3. NGBoost

df_train = df_train.dropna()
df_test = df_test.dropna()

df_train['AIR_PRESSURE'] = (lambda ap : (ap-900) / 200)(df_train['AIR_PRESSURE'])
df_test['AIR_PRESSURE'] = (lambda ap : (ap-900) / 200)(df_test['AIR_PRESSURE'])
df_train['AIR_TEMPERATURE'] = (lambda at : (at+50) / 120)(df_train['AIR_TEMPERATURE'])
df_test['AIR_TEMPERATURE'] = (lambda at : (at+50) / 120)(df_test['AIR_TEMPERATURE'])
df_train['day_min'] = (lambda dm : dm / 1439)(df_train['day_min'])
df_test['day_min'] = (lambda dm : dm / 1439)(df_test['day_min'])
df_train['HUMIDITY'] = (lambda h : h / 100)(df_train['HUMIDITY'])
df_test['HUMIDITY'] = (lambda h : h / 100)(df_test['HUMIDITY'])
df_train['WIND_SPEED'] = (lambda ws : ws / 80)(df_train['WIND_SPEED'])
df_test['WIND_SPEED'] = (lambda ws : ws / 80)(df_test['WIND_SPEED'])

feature_cols = ['AIR_TEMPERATURE', 'AIR_PRESSURE','day_min', 'HUMIDITY', 'WIND_SPEED']
label_cols = ['AIR_TEMPERATURE']

y_train = df_train[label_cols].values
X_train = df_train[feature_cols].values
y_test = df_test[label_cols].values
X_test = df_test[feature_cols].values

X_train = X_train[:-1]
y_train = y_train[1:]
X_test = X_test[:-1]
y_test = y_test[1:]

plt.scatter(np.arange(0, len(df_test)), df_test.AIR_TEMPERATURE)
plt.savefig("abnomal.png")

models = os.listdir('./NGBoost_model')
#os.mkdir('./spike_abnomals')
#os.mkdir('./After_NG')

for i in models:
    model_ngb = joblib.load('./{0}.pkl'.format(i))
    # model_ngb.fit(X_train, y_train, early_stopping_rounds = 5)

    y_train_ngb = model_ngb.pred_dist(X_train)
    abnormals = []

    dum = np.ones((100, 5)) * X_test[0][:]
    X_test = np.concatenate([dum, X_test], axis=0)

    raw_x = X_test.copy()
    ab_count = 0
    go_count = 0
    for i in range(1438):
        temp = model_ngb.pred_dist(np.expand_dims(X_test[i - 1][:], axis=0))
        t_un = temp.dist.interval(0.1)[0]
        t_up = temp.dist.interval(0.1)[1]

        if ab_count >= 1:
            ab_count = 0
            go_count = 0
        elif go_count >= 1:
            go_count = 0
            ab_count = 0
        else:
            if (X_test[i][0] >= t_up[0]) or (X_test[i][0] <= t_un[0]):
                X_test[i][0] = temp.loc[0]
                abnormals.append(i)
                ab_count += 1
            else:
                go_count += 1

    for i in range(1438):
        temp = model_ngb.pred_dist(np.expand_dims(X_test[i - 1][:], axis=0))
        t_un = temp.dist.interval(0.3)[0]
        t_up = temp.dist.interval(0.3)[1]
        if (X_test[i][0] >= t_up[0]) or (X_test[i][0] <= t_un[0]):
            X_test[i][0] = temp.loc[0]
            abnormals.append(i)
            ab_count += 1

    abnormals = set(abnormals)
    abnormals = list(abnormals)
    print(abnormals)
    print(len(abnormals))

    # X_test[i][1] = model_ngb.predict(np.expand_dims(X_test[i - 1][:], axis=0))
    # print(((X_test[i][1]) * 120) - 50)

    y_test_ngb = model_ngb.pred_dist(X_test)

    predictions = pd.DataFrame(y_test_ngb.loc, columns=['Predictions'])
    predictions_upper = pd.DataFrame(y_test_ngb.dist.interval(0.2)[1], columns=['Predictions_upper'])
    predictions_lower = pd.DataFrame(y_test_ngb.dist.interval(0.2)[0], columns=['Predictions_lower'])
    # df_test = df_test[['COLCT_DT','AIR_TEMPERATURE']]
    df_test = pd.DataFrame(raw_x[:, 0], columns=['AIR_TEMPERATURE'])

    df_figure = pd.concat([df_test, predictions, predictions_lower, predictions_upper], axis=1)

    df_figure[['AIR_TEMPERATURE', 'Predictions', 'Predictions_lower', 'Predictions_upper']] = (
        lambda at: (at * 120) - 50)(
        df_figure[['AIR_TEMPERATURE', 'Predictions', 'Predictions_lower', 'Predictions_upper']])


    def plot_results(df, title):
        fig, ax = plt.subplots(figsize=(22, 5))
        x_axis = np.arange(len(df))
        plt.ylim([5, 35])
        plt.plot(x_axis, df.Predictions, label='Air_Temperature_Predicted', color='b', lw=2)
        plt.fill_between(x_axis, df.Predictions_lower, df.Predictions_upper, label='30% after 10% Prediction Interval',
                         color='gray', alpha=0.5)
        plt.scatter(x_axis, df['AIR_TEMPERATURE'], label='AIR_TEMPERATURE Actual', color='g', lw=1)
        ax.legend(fontsize=14)
        plt.title('Actual vs NGBoost Predicted')
        plt.xlabel(title)
        plt.show()


    df_figure_fw = df_figure
    # df_figure_fw
    print(df_figure)
    plot_results(df_figure_fw, 'Last Day in Test Set')

    wrong = []
    real_wrong = []
    for i in range(1438):
        if (df_figure.AIR_TEMPERATURE[i] >= df_figure.Predictions_upper[i]) or (
                df_figure.AIR_TEMPERATURE[i] <= df_figure.Predictions_lower[i]):
            wrong.append(i)
        if (df_figure.AIR_TEMPERATURE[i] < 20.9) or (df_figure.AIR_TEMPERATURE[i] > 23.6):
            real_wrong.append(i)

    print(wrong)
    print(len(wrong))
    print(real_wrong)
    print(len(real_wrong))



    df_figure.to_csv('C:/Users/USER/Desktop/준익의 흔적/NGBoost.csv', index=False)

#df = df_figure
#df.AIR_TEMPERATURE = df.AIR_TEMPERATURE[(df['AIR_TEMPERATURE'] <= df['Predictions_upper']) and (df['AIR_TEMPERATURE'] >= df['Predictions_lower'])]
#df.info()

# 4. 기존 방법론
"""
df = df_test
print("기존 NULL 갯수 : ",df['AIR_TEMPERATURE'].isnull().sum().sum())

df[df['AIR_TEMPERATURE'] < -50] = np.nan
df[df['AIR_TEMPERATURE'] > 70 ] = np.nan

print("유효범위 NULL 갯수 : ",df['AIR_TEMPERATURE'].isnull().sum().sum())

df['AIR_TEMPERATURE_delta'] = abs(df['AIR_TEMPERATURE']-df['AIR_TEMPERATURE'].shift(1))

df[df['AIR_TEMPERATURE_delta']>(df['AIR_TEMPERATURE_delta'].mean()*3)] = np.nan

df_plus = pd.DataFrame({'AIR_TEMPERATURE_upper': df.AIR_TEMPERATURE+df.AIR_TEMPERATURE_delta.mean()*3,
                        'AIR_TEMPERATURE_lower': df.AIR_TEMPERATURE-df.AIR_TEMPERATURE_delta.mean()*3})


print("최종 NULL 갯수 : ",df['AIR_TEMPERATURE'].isnull().sum().sum())
df = pd.concat([df,df_plus],axis=1)

plt.figure(figsize=(22, 5))
plt.ylim([20, 25])
x_axis = np.arange(len(df))
plt.scatter(x_axis, df.AIR_TEMPERATURE, c='green',label='AIR_TEMPERATURE' ,lw=1)
plt.fill_between(x_axis, df.AIR_TEMPERATURE_lower,  df.AIR_TEMPERATURE_upper,label = 'dleta_average*3', color='gray', alpha=0.5)
plt.legend(fontsize=14)
plt.show()
"""