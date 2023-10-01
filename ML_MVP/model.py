#!/usr/bin/env python
# coding: utf-8
import pickle

# In[102]:


def forecast(subm_data, train_data, holidays):
# 0. ЗАГРУЖАЕМ МОДЕЛИ
    model_lgbmr_1 = pickle.load(open('model_lgbmr_1.pkl', 'rb'))
    model_lgbmr_2_b = pickle.load(open('model_lgbmr_2_b.pkl', 'rb'))
    model_lgbmr_2_t = pickle.load(open('model_lgbmr_2_t.pkl', 'rb'))
    model_lgbmr_4 = pickle.load(open('model_lgbmr_4.pkl', 'rb'))

# 1. ФУНКЦИИ ПОДГОТОВКИ ФИЧЕЙ:
    # для добавления выходных
    def add_holidays(df, rus_holidays):
        rus_holidays['date'] = pd.to_datetime(rus_holidays['date'], format='%d.%m.%Y')
        df = df.merge(rus_holidays[['date', 'holiday']], on='date')
        df = df.set_index('date')
        df.holiday = df.holiday.astype(str)
        return df

    # для разделения на группы
    def df_divide(df):
    # Четыре основные группы, получены анализом матрицы корреляции
        pr_group_list_id = ['1ff1de774005f8da13f42943881c655f',
                            '32bb90e8976aab5298d5da10fe66f21d',
                            '3c59dc048e8850243be8079a5c74d079',
                            '98f13708210194c475687be6106a3b84',
                            '6512bd43d9caa6e02c990b0a82652dca',
                            'aab3238922bcc25a6f606eb525ffdc56',
                            'c74d97b01eae257e44aa9d5bade97baf',
                            'c20ad4d76fe97759aa27a0c99bff6710',
                            'c51ce410c124a10e0db5e4b97fc2af39']
    # Группа 1
        group_1 = df.loc[((df.st_type_loc_id == '1') | (df.st_type_loc_id == '2'))
                        &((df.pr_group_id == pr_group_list_id[0])|
                          (df.pr_group_id == pr_group_list_id[1])|
                          (df.pr_group_id == pr_group_list_id[2])|
                          (df.pr_group_id == pr_group_list_id[3]))]   
    # Группа 2
        group_2_full = df.loc[((df.st_type_loc_id=='1') | (df.st_type_loc_id=='2'))
                        &((df.pr_group_id == pr_group_list_id[4])|
                          (df.pr_group_id == pr_group_list_id[5])|
                          (df.pr_group_id == pr_group_list_id[6])|
                          (df.pr_group_id == pr_group_list_id[7])|
                          (df.pr_group_id == pr_group_list_id[8]))]
    # Группа 3
        group_3 = df.loc[(df.st_type_loc_id=='3')
                        &((df.pr_group_id == pr_group_list_id[0])|
                          (df.pr_group_id == pr_group_list_id[1])|
                          (df.pr_group_id == pr_group_list_id[2])|
                          (df.pr_group_id == pr_group_list_id[3]))]
    # Группа 4
        group_4 = df.loc[(df.st_type_loc_id=='3')
                        &((df.pr_group_id == pr_group_list_id[4])|
                          (df.pr_group_id == pr_group_list_id[5])|
                          (df.pr_group_id == pr_group_list_id[6])|
                          (df.pr_group_id == pr_group_list_id[7])|
                          (df.pr_group_id == pr_group_list_id[8]))]
    
    # Разделение группы 2 на две ментшие группы на основании суммарного количества проданных товаров
    # Рейтинг всех товаров по количеству проданных штук:
        full_pr_sku = pd.DataFrame(group_2_full.groupby('pr_sku_id')
                               .agg({'pr_sales_in_units':'sum'})
                               .sort_values(by='pr_sales_in_units', ascending=False)
                               .reset_index().pr_sku_id)
    
    # временный признак для будущего разделения 25:75:
        full_pr_sku['temp_label'] = 0
        full_pr_sku.loc[full_pr_sku.index<len(full_pr_sku)//4, 'temp_label'] = 1
    
    # разметка второй гуппы
        groups_labeled = pd.merge(group_2_full,
                              full_pr_sku, 
                              how='left', 
                              on='pr_sku_id')
    
    # Группа 2 TOP (товары с наибольшими продажами, первые 25% в рейтинге)
        groups_labeled.set_index(group_2_full.index,inplace=True)
        group_2_top = groups_labeled.loc[groups_labeled.temp_label==1]
        group_2_top = group_2_top.drop('temp_label',axis=1)
    
    # Группа 2 BOTTOM (оставшиеся товары)
        group_2_bottom = groups_labeled.loc[groups_labeled.temp_label==0]
        group_2_bottom = group_2_bottom.drop('temp_label',axis=1)
    
        return group_1, group_2_top, group_2_bottom, group_3, group_4
    
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
    def split_and_prophet(df,split_date):
# делим выборку
        train = df.loc[df.index < pd.to_datetime(split_date, format='%Y-%m-%d')]
        test = df.loc[df.index >= pd.to_datetime(split_date, format='%Y-%m-%d')]
    
# признаки prophet для обучающей выборки

# Prophet по штукам
        train_prophet_unit = train.copy()
        train_prophet_unit = train_prophet_unit.pr_sales_in_units.reset_index()
        train_prophet_unit = train_prophet_unit.rename(columns={'date': 'ds', 'pr_sales_in_units': 'y'})
        model_prophet_unit = Prophet()
        model_prophet_unit.fit(train_prophet_unit)
        future_unit = model_prophet_unit.make_future_dataframe(periods=(len(test.index.unique())))
        forecast_unit = model_prophet_unit.predict(future_unit)
        prophet_data_unit = forecast_unit[['trend','yhat','weekly','ds']].set_index('ds', drop=True)
    
# Prophet по рублям
        train_prophet_rub = train.copy()
        train_prophet_rub = train_prophet_rub.pr_sales_in_rub.reset_index()
        train_prophet_rub = train_prophet_rub.rename(columns={'date': 'ds', 'pr_sales_in_rub': 'y'})
        model_prophet_rub = Prophet()
        model_prophet_rub.fit(train_prophet_rub)
        future_rub = model_prophet_rub.make_future_dataframe(periods=(len(test.index.unique())))
        forecast_rub = model_prophet_rub.predict(future_rub)
        prophet_data_rub = forecast_rub[['trend','yhat','weekly','ds']].set_index('ds', drop=True)
        prophet_data_rub = prophet_data_rub.rename(columns={'trend':'trend_rub', 'yhat':'yhat_rub','weekly':'weekly_rub'})

# добавляем признаки от prophet
        train = pd.merge(train,prophet_data_unit, how='left', left_index=True, right_index=True)
        train = pd.merge(train,prophet_data_rub, how='left', left_index=True, right_index=True)
        feature_train = train.drop(['pr_sales_in_units','pr_sales_in_rub'], axis=1)
        target_train = train.pr_sales_in_units
# тестовая выборка    
        test = pd.merge(test,prophet_data_unit, how='left', left_index=True, right_index=True)
        test = pd.merge(test,prophet_data_rub, how='left', left_index=True, right_index=True)
        features_test = test.drop(['pr_sales_in_units','pr_sales_in_rub'], axis=1)
        target_test = test.pr_sales_in_units
        return feature_train, target_train,features_test,target_test    


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
        gr_1,gr_2_top,gr_2_bot,gr_3, gr_4 = df_divide(df)
    # добавляем лаги
        gr_1 = make_features(gr_1, 14, 14)
        gr_2_top = make_features(gr_2_top, 14, 14)
        gr_2_bot = make_features(gr_2_bot, 14, 14)
        gr_4 = make_features(gr_4, 14, 14)
    # заполняем пропуски от лагов
        gr_1 = gr_1.fillna(0)
        gr_2_top = gr_2_top.fillna(0)
        gr_2_bot = gr_2_bot.fillna(0)
        gr_4 = gr_4.fillna(0)
    # добавляем признаки Prophet по штукам и рублям, выделяем test_features:
        feature_train_gr_1, target_train_gr_1, features_test_gr_1, target_test_gr_1 = split_and_prophet(gr_1,first_test_date)
        feature_train_gr_2_t, target_train_gr_2_t, features_test_gr_2_t, target_test_gr_2_t = split_and_prophet(gr_2_top,first_test_date)
        feature_train_gr_2_b, target_train_gr_2_b, features_test_gr_2_b, target_test_gr_2_b = split_and_prophet(gr_2_bot,first_test_date)
        feature_train_gr_4, target_train_gr_4, features_test_gr_4, target_test_gr_4 = split_and_prophet(gr_4,first_test_date)
        return features_test_gr_1, features_test_gr_2_t, features_test_gr_2_b, features_test_gr_4

    # ДЕЛАЕМ ПРЕДСКАЗАНИЕ
    def get_prediction(
        features_test_gr_1,
        features_test_gr_2_t,
        features_test_gr_2_b,
        features_test_gr_4,
    ):
        prediction_1 = model_lgbmr_1.predict(features_test_gr_1)
        prediction_2_t = model_lgbmr_2_t.predict(features_test_gr_2_t)
        prediction_2_b = model_lgbmr_2_b.predict(features_test_gr_2_b)
        prediction_4 = model_lgbmr_4.predict(features_test_gr_4)
        return prediction_1, prediction_2_t, prediction_2_b, prediction_4
    
# СОБИРАЕМ ПРЕДСКАЗАНИЯ В SALES_SUBMISSION
    def predictions_to_submission(
        sub_data,
        prediction_1,
        prediction_2_t,
        prediction_2_b,
        prediction_4,
        features_test_gr_1,
        features_test_gr_2_t,
        features_test_gr_2_b,
        features_test_gr_4
    ):
    # колонка для корректного merge    
        sub_data['for_merge'] = sub_data['st_id']+sub_data['pr_sku_id']+sub_data['date']
    # удаляем нулевую колонку target
        sub_data = sub_data.drop('target', axis=1)
    # готовим группы к merge
    # добавляем полученные target
        features_test_gr_1['target'] = prediction_1
        features_test_gr_2_t['target'] = prediction_2_t
        features_test_gr_2_b['target'] = prediction_2_b
        features_test_gr_4['target'] = prediction_4
    # колонка для корректного merge
        features_test_gr_1['for_merge']   = (features_test_gr_1['st_id']
                                         +features_test_gr_1['pr_sku_id']
                                         +features_test_gr_1.index.astype(str))
        features_test_gr_2_t['for_merge'] = (features_test_gr_2_t['st_id']
                                         +features_test_gr_2_t['pr_sku_id']
                                         +features_test_gr_2_t.index.astype(str))
        features_test_gr_2_b['for_merge'] = (features_test_gr_2_b['st_id']
                                         +features_test_gr_2_b['pr_sku_id']
                                         +features_test_gr_2_b.index.astype(str))
        features_test_gr_4['for_merge']   = (features_test_gr_4['st_id']
                                         +features_test_gr_4['pr_sku_id']
                                         +features_test_gr_4.index.astype(str))
        full_set = pd.concat([features_test_gr_1,
                          features_test_gr_2_t,
                          features_test_gr_2_b,
                          features_test_gr_4])
        full_set = full_set[['for_merge','target']]
    
    # делаем submission
        sub_data = pd.merge(sub_data, full_set, how='left', on='for_merge')
    # удаляем вспомогательные данные
        sub_data = sub_data.drop('for_merge', axis=1)
        return sub_data    
    
# 3.ДЕЙСТВИЯ:
    # объединяем submission с трейн
    test_data, split, sales_sub = get_sub_df(subm_data, train_data)
    # готовим тестовые выборки
    ft_1, ft_2_t, ft_2_b, ft_4 = get_test_features(test_data,split)
    # получаем предсказания
    pr_1, pr_2_t, pr_2_b, pr_4 = get_prediction(ft_1, ft_2_t, ft_2_b, ft_4)
    # объединяем предсказания в submission датафрейм
    sales_submission_pred = predictions_to_submission(sales_sub,pr_1, pr_2_t, pr_2_b, pr_4, ft_1, ft_2_t, ft_2_b, ft_4)
    return sales_submission_pred


# In[ ]:





# In[ ]:





# In[ ]:




