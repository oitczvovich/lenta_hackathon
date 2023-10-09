import numpy as np
import pandas as pd

from copy import deepcopy
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#from lightgbm import LGBMRegressor 
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

import joblib
from hybridclass import HybridModel


def forecast(subm_data, train_data, holidays):
    # 0. ЗАГРУЖАЕМ МОДЕЛИ
    model_prophet_unit_1 = joblib.load('model_prophet_unit_1.pkl')
    model_prophet_rub_1 = joblib.load('model_prophet_rub_1.pkl')
    model_prophet_unit_2 = joblib.load('model_prophet_unit_2.pkl')
    model_prophet_rub_2 = joblib.load('model_prophet_rub_2.pkl')
    model_gr_1 = joblib.load('model_gr_1.pkl')
    model_gr_2 = joblib.load('model_gr_2.pkl')
    
# 1. ФУНКЦИИ ПОДГОТОВКИ ФИЧЕЙ:
    # для добавления выходных
    def add_holidays(df,rus_holidays):
        rus_holidays['date'] = pd.to_datetime(rus_holidays['date'], format='%d.%m.%Y')
        df = df.merge(rus_holidays[['date', 'holiday']], on='date')
        df = df.set_index('date')
        df['holiday'] = df['holiday'].astype(str)
        return df
    
    # для разделения на группы
    def df_divide_rate(df):
    # ТОП-товары
        top_sku = (df.groupby('pr_sku_id')
                   .agg({'pr_sales_in_units':'sum'})
                   .sort_values(by='pr_sales_in_units', ascending=False)
                   .head(50)
                   .index
                   .tolist())
    # Группа 1
        group_1 = df.loc[df['pr_sku_id'].isin(top_sku)]
  
    # Группа 2
        group_2 = df.loc[~df['pr_sku_id'].isin(top_sku)]

    
        return group_1, group_2    
    
    # для генерации лагов
    def make_features(data, max_lag, rolling_mean_size):
        data['day'] = data.index.day.astype(str)
        data['dayofweek'] = data.index.dayofweek.astype(str)
        data['month'] = data.index.month.astype(str)
    
    # Создаем лаги для продаж для каждой комбинации 'st_id', 'pr_sku_id' и 'pr_sales_type_id'
        for lag in range(1, max_lag + 1):
            data[f'lag_{lag}_pr_sales_in_units'] = data.groupby(
                ['st_id','pr_sku_id','pr_sales_type_id'])['pr_sales_in_units'].shift(lag)

        data['rolling_mean'] = data['pr_sales_in_units'].shift().rolling(rolling_mean_size).mean()
        return data    
    
    # для генерации фичей и деления выделения тестовой выборки
    def split_and_prophet(df,split_date, prophet_unit, prophet_rub):
    # делим выборку
        train = df.loc[df.index < pd.to_datetime(split_date, format='%Y-%m-%d')]
        test = df.loc[df.index >= pd.to_datetime(split_date, format='%Y-%m-%d')]

    # Prophet по штукам
        future_unit = prophet_unit.make_future_dataframe(periods=(len(test)))
        forecast_unit = prophet_unit.predict(future_unit)
        prophet_data_unit = forecast_unit[['trend','yhat','weekly','ds']].set_index('ds', drop=True)
    
    # Prophet по рублям
        future_rub = prophet_rub.make_future_dataframe(periods=(len(test)))
        forecast_rub = prophet_rub.predict(future_rub)
        prophet_data_rub = forecast_rub[['trend','yhat','weekly','ds']].set_index('ds', drop=True)
        prophet_data_rub = prophet_data_rub.rename(columns={'trend':'trend_rub', 'yhat':'yhat_rub','weekly':'weekly_rub'})

    # добавляем признаки от prophet  
        test = pd.merge(test,prophet_data_unit, how='left', left_index=True, right_index=True)
        test = pd.merge(test,prophet_data_rub, how='left', left_index=True, right_index=True)
        features_test = test.drop(['pr_sales_in_units','pr_sales_in_rub'], axis=1)
        target_test = test[['pr_sales_in_units']]
        return features_test,target_test    


    # 2. ФУНКЦИИ ОСНОЫНЫЕ:
    # ОБЪЕДИНЕНИЕ SUBMISSION С ОБУЧАЮЩЕЙ ВЫБОРКОЙ:
    def get_sub_df(sub_df, train_df):
    # удаление товаров из submission, которых нет в train
        sub_df = sub_df.loc[sub_df.pr_sku_id.isin(train_df.pr_sku_id.unique().tolist())]
        sub_df_origin = sub_df.copy()
    # признаки для включения в submission
    # 'st_type_loc_id'
        loc_info = train_df.groupby('st_id').agg({'st_type_loc_id':'unique'}).reset_index()
        loc_info['st_type_loc_id'] = loc_info['st_type_loc_id'].apply(''.join)
    # 'pr_group_id'
        group_info = train_df.groupby('pr_sku_id')['pr_group_id'].unique().reset_index()
        group_info['pr_group_id'] = group_info['pr_group_id'].apply(''.join)
    # merge признаков к submission
        sub_df = pd.merge(sub_df,loc_info, how='left', on='st_id')
        sub_df = pd.merge(sub_df,group_info, how='left', on='pr_sku_id')
    # подготовка индексов перед merge
        sub_df['date'] = pd.to_datetime(sub_df['date'], format='%Y-%m-%d')
        sub_df = sub_df.set_index('date', drop=True)
        sub_df.sort_index(inplace=True)
    # сохраняем дату начала прогноза
        split_date = sub_df.index[0]
    # собираем в единый датафрейм
        train_sub_df = pd.concat([train_df,sub_df])
    # обработка пропусков
        train_sub_df = train_sub_df.drop('target', axis=1)
        train_sub_df['pr_sales_type_id'] = '0'
        train_sub_df.pr_sales_in_units = train_sub_df.pr_sales_in_units.fillna(0)
        train_sub_df.pr_sales_in_rub = train_sub_df.pr_sales_in_rub.fillna(0)
        train_sub_df = train_sub_df.fillna('drop_value') 
        return train_sub_df, split_date, sub_df_origin
    
    # ПОДГОТОВКА ТЕСТОВОЙ ВЫБОРКИ РАЗБИТОЙ НА ГРУППЫ:
    def get_test_features(df, first_test_date):
    # добавляем флаг выходного дня
        df = add_holidays(df, holidays)
    # делим на группы
        gr_1,gr_2_top = df_divide_rate(df)
    # добавляем лаги
        gr_1 = make_features(gr_1, 14, 14)
        gr_2_top = make_features(gr_2_top, 14, 14)
    # заполняем пропуски от лагов
        gr_1 = gr_1.fillna(0)
        gr_2_top = gr_2_top.fillna(0)
    # добавляем признаки Prophet по штукам и рублям, выделяем test_features:
        features_test_gr_1, target_test_gr_1 = split_and_prophet(gr_1,first_test_date, model_prophet_unit_1, model_prophet_rub_1)
        features_test_gr_2_t, target_test_gr_2_t = split_and_prophet(gr_2_top,first_test_date, model_prophet_unit_2, model_prophet_rub_2)
        return features_test_gr_1, features_test_gr_2_t
    
    # ДЕЛАЕМ ПРЕДСКАЗАНИЕ
    def get_prediction(features_test_gr_1,
                       features_test_gr_2_t):
        prediction_1   = model_gr_1.predict(features_test_gr_1)
        prediction_2_t = model_gr_2.predict(features_test_gr_2_t)
        return prediction_1, prediction_2_t
    
    # СОБИРАЕМ ПРЕДСКАЗАНИЯ В SALES_SUBMISSION
    def predictions_to_submission(sub_data,
                              prediction_1,
                              prediction_2_t,
                              features_test_gr_1,
                              features_test_gr_2_t):
    # колонка для корректного merge    
        sub_data['for_merge'] = sub_data['st_id']+sub_data['pr_sku_id']+sub_data['date']
    # удаляем нулевую колонку target
        sub_data = sub_data.drop('target', axis=1)
    # готовим группы к merge
    # добавляем полученные target
        features_test_gr_1['target'] = prediction_1
        features_test_gr_2_t['target'] = prediction_2_t
    # колонка для корректного merge
        features_test_gr_1['for_merge']   = (features_test_gr_1['st_id']
                                         +features_test_gr_1['pr_sku_id']
                                         +features_test_gr_1.index.astype(str))
        features_test_gr_2_t['for_merge'] = (features_test_gr_2_t['st_id']
                                         +features_test_gr_2_t['pr_sku_id']
                                         +features_test_gr_2_t.index.astype(str))

        full_set = pd.concat([features_test_gr_1,
                          features_test_gr_2_t])
        full_set = full_set[['for_merge','target']]
        
        full_set['target'] = full_set['target'].astype(int)
    
    # делаем submission
        sub_data = pd.merge(sub_data, full_set, how='left', on='for_merge')
    # удаляем вспомогательные данные
        sub_data = sub_data.drop('for_merge', axis=1)
        return sub_data    
    
    # 3.ДЕЙСТВИЯ:
    # объединяем submission с трейн
    test_data, split, sales_sub = get_sub_df(subm_data, train_data)
    # готовим тестовые выборки
    ft_1, ft_2_t = get_test_features(test_data,split)
    # получаем предсказания
    pr_1, pr_2_t = get_prediction(ft_1, ft_2_t)
    # объединяем предсказания в submission датафрейм
    sales_submission_pred = predictions_to_submission(sales_sub,
                                                      pr_1, 
                                                      pr_2_t, 
                                                      ft_1, 
                                                      ft_2_t)
    return sales_submission_pred