import requests
import os
# import logging
from datetime import date, timedelta
import json
import pandas as pd
import csv
import time
from dotenv import load_dotenv

from ml_model.model import forecast
from core.settings import settings
from core.logger import setup_logging, _logger
# from core.func_API import (
#     get_categs_info,
#     get_sales,
#     get_stores,
#     get_token
# )


load_dotenv()
current_direct = os.path.join(os.getcwd(), 'out_data')


NOW_DATE=date.today()
PREDICTION_DAYS: int = 14


URL_CATEGORIES = "categories"
URL_SALES = "sales"
URL_STORES = "shops"
URL_FORECAST = "forecast"
URL_LOGIN = "login"

api_port = settings.api_port
api_host = settings.api_host
name_user = settings.name_user
password_user = settings.password_user


predict = os.path.join(os.getcwd(), 'predict.csv')




def get_address(resource):
    return "http://" + api_host + ":" + api_port + "/" + resource


# функции запросы к БД.
def get_stores():
    """ Запрос для получения списка магазинов.
    """
    stores_url = get_address(URL_STORES)
    resp = requests.get(stores_url, headers=headers)
    if resp.status_code != 200:
        _logger.warning("Could not get stores list")
        return [], resp.status_code
    return resp.json()["data"]
    
def get_sales(store_id=None, product_id=None):
    """
    """
    sale_url = get_address(URL_SALES)
    params = {}
    if store_id is not None:
        params["store_id"] = store_id
    if product_id is not None:
        params["product_id"] = product_id
    resp = requests.get(sale_url, params=params, headers=headers)
    if resp.status_code != 200:
        _logger.warning("Could not get sales history")
        return []

    return resp.json()["data"]

def get_categs_info():
    """

    """
    categs_url = get_address(URL_CATEGORIES)
    resp = requests.get(categs_url, headers=headers)
    if resp.status_code != 200:
        _logger.warning("Could not get category info")
        return {}
    result = {el["pr_sku_id"]: el for el in resp.json()["data"]}
    return result

def get_token():
    """Получить токен для обращения к API."""
    url_login = get_address(URL_LOGIN)
    user_data = {"username": name_user, "password": password_user}
    r = requests.post(url_login, data=user_data)

    if r.status_code == 200:
        token = r.json().get('access')
        return token
    else:
        return None


def write_json(val, name):
    with open(f"{name}.json", "w") as f:
        json.dump(val, f)

def create_csv(data_dict: dict, name_file: str):
    """ Создание CSV файлв и из словаря."""
    fieldnames = data_dict.keys()
    file_name = f'{name_file}.csv'
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(data_dict)


# Функции для подготовки данных.
def create_predict(sales, item_info, store):
    """Подготовка данных predict.
    создвется DataFrame
    """
    DF_predict = pd.DataFrame()
    st_pr_data_dict = {
        'pr_sku_id': item_info['pr_sku_id'],
        'pr_group_id': item_info['pr_group_id'],
        'pr_cat_id': item_info['pr_cat_id'],
        'pr_subcat_id': item_info['pr_subcat_id'],
        'pr_uom_id': item_info['pr_uom_id'],
        'st_id': store['st_id'],
        'st_is_active': store['st_is_active'],
        'st_city_id': store['st_city_id'],
        'st_division_code': store['st_division_code'],
        'st_type_format_id': store['st_type_format_id'],
        'st_type_loc_id': store['st_type_loc_id'],
        'st_type_size_id': store['st_type_size_id'],
    }

    for sales_pr in sales:
        data_dict = {
            'date': sales_pr['date'],
            'pr_sales_type_id': sales_pr['pr_sales_type_id'],
            'pr_sales_in_units': sales_pr['pr_sales_in_units'],
            'pr_sales_in_rub': sales_pr['pr_sales_in_rub'],
        }
        st_pr_data_dict.update(data_dict)
        DF_predict = DF_predict._append(data_dict, ignore_index=True)
        DF_predict['date'] = pd.to_datetime(DF_predict['date'], format='%Y-%m-%d')
    return DF_predict


def create_subm_DF():
    """ Подготовка данных для предсказания.
    st_id – захэшированное id магазина;
    pr_sku_id – захэшированное id товара;
    date – дата (день) текущий день + 14 дней.
    Возвращается DataFrame.
    """
    DF_subm_data = pd.DataFrame()
    path_csv_subm_data = os.path.join(os.getcwd(), 'dict_subm.csv')
    DF_subm_data = pd.read_csv(path_csv_subm_data)
    
        # DF_subm_data = DF_subm_data._append(data_dict, ignore_index=True)
    DF_subm_data['date'] = pd.to_datetime(DF_subm_data['date'], format='%Y-%m-%d')
        # DF_subm_data.info()
    return DF_subm_data


def create_holidays():
    """ Создание Dataframe из файла holidays_covid_calendar_2
    Обработка только поля date, т.к. другие данные не нужны.
    """
    holidays_path = os.path.join(os.getcwd(), 'holidays_covid_calendar_2.csv')
    holidays_df = pd.read_csv(holidays_path)
    return holidays_df


def create_train_DF():
    """ Подготовка данных для предсказания.  
    В train_data:  
    st_id – захэшированное id магазина;  
    pr_sku_id – захэшированное id товара;  
    date – дата sales;  
    pr_sales_type_id – флаг наличия промо;  
    pr_sales_in_units – число проданных товаров без признака промо;  
    pr_sales_in_rub – продажи без признака промо в РУБ;
    Возвращаються данные.
                               st_id                         pr_sku_id  st_type_loc_id        date  pr_sales_type_id  pr_sales_in_units  pr_sales_in_rub
0   1aa057313c28fa4a40c5bc084b11d276  fd064933250b0bfe4f926b867b0a5ec8               3  2022-08-24                 0                  4             98.0
1   1aa057313c28fa4a40c5bc084b11d276  fd064933250b0bfe4f926b867b0a5ec8               3  2022-08-08                 0                  3             73.0        """
    
    # здесь пытался сделать описание колонок, но появлялась ошибка пока все закоментил.
    DF_train_data = pd.DataFrame()
    
    path_train_data = os.path.join(os.getcwd(), 'dict_train.csv')
    DF_train_data = pd.read_csv(path_train_data)

        # DF_train_data = DF_train_data._append(st_pr_data_dict, ignore_index=True)
    # данные "date" к типу данных datetime
    DF_train_data['date'] = pd.to_datetime(DF_train_data['date'], format='%Y-%m-%d')
    DF_train_data = DF_train_data.set_index('date', drop=True)
    
    # расположение данных по порядку
    DF_train_data.sort_index(inplace=True)
    
    for column in [
        'st_type_format_id',
        'st_type_loc_id',
        'st_type_size_id', 
        'pr_uom_id',
        'pr_sales_type_id'
    ]:
        DF_train_data[column] = DF_train_data[column].astype(str)
    # изменение последовательности колонок        
    DF_train_data = DF_train_data[[ 
             'st_id', 
             'st_city_id', 
             'st_division_code',
             'st_type_format_id',
             'st_type_loc_id',
             'st_type_size_id',
             'pr_sku_id',
             'pr_group_id',
             'pr_cat_id', 
             'pr_subcat_id',
             'pr_uom_id',
             'pr_sales_type_id',
             'pr_sales_in_units',
             'pr_sales_in_rub',
            ]]

    # DF_train_data.info()
    return DF_train_data



def create_dict_subm_data(item_info, store, forecast_dates):
    data_dict = []
    for forecast_day in forecast_dates:
        print('forecast_day', forecast_day, type(forecast_day))
        data_dict = {
            'st_id': store['st_id'],
            'pr_sku_id': item_info['pr_sku_id'],
            'date': forecast_day,
            'target': 0,
        }
        create_csv(data_dict, 'dict_subm')
    return data_dict

def create_dict_train(sales, item_info, store):
    st_pr_data_dict = {
        'st_id': store['st_id'],
        'st_city_id': store['st_city_id'],
        'st_division_code': store['st_division_code'],
        'st_type_format_id': store['st_type_format_id'],
        'st_type_loc_id': store['st_type_loc_id'],
        'st_type_size_id': store['st_type_size_id'],
        'pr_sku_id': item_info['pr_sku_id'],
        'pr_group_id': item_info['pr_group_id'],
        'pr_cat_id': item_info['pr_cat_id'],
        'pr_subcat_id': item_info['pr_subcat_id'],
        'pr_uom_id': str(item_info['pr_uom_id']),
    }

    for sales_pr in sales:
        data_dict = {
            'date':  sales_pr['date'],
            'pr_sales_type_id': sales_pr['pr_sales_type_id'],
            'pr_sales_in_units': float(sales_pr['pr_sales_in_units']),
            'pr_sales_in_rub': sales_pr['pr_sales_in_rub'],
        }

        st_pr_data_dict.update(data_dict)
        create_csv(st_pr_data_dict, 'dict_train')
    return st_pr_data_dict                     


def _create_predata_csv(forecast_dates):
    _logger.info('Запуск подготовки данных')
    categs_info = get_categs_info()  # список product_id
    # закоммитил для тестоваs
    for store in get_stores():
        store_id = store["st_id"]
        for product_id in get_categs_info():
            print('product_id', product_id)
            result = []
            predict = []
            # получаем id продукта
            for item in get_sales(store_id=store_id, product_id=product_id):
                item_info = categs_info[item["product_id"]]
                sales = item["fact"]
                # создание csv магазин/продукт/дата_предсказания/таргет
                create_dict_subm_data(item_info, store, forecast_dates)
                # создание данных по продажам
                create_dict_train(sales, item_info, store)


def _create_df_from_csv():
    DF_subm_data = create_subm_DF()
    # DF_subm_data.info()
    DF_train_data = create_train_DF()
    # DF_train_data.info()

    return DF_subm_data, DF_subm_data

def main(today=date.today()):
    # forecast_dates = [NOW_DATE + timedelta(days=d) for d in range(1, PREDICTION_DAYS + 1)]
    # forecast_dates = [el.strftime('%Y-%m-%d') for el in forecast_dates]
    # _create_predata_csv(forecast_dates=forecast_dates)

    DF_subm_data, DF_train_data = _create_df_from_csv()

    print("Запуск формирования предсказания.")
    holidays = create_holidays()
    forecast(subm_data=DF_subm_data, train_data=DF_train_data, holidays=holidays)
                
                
                
            #     prediction = forecast(sales, item_info, store)
                
            #     result.append({"store": store["store"],
            #                 "forecast_date": today.strftime("%Y-%m-%d"),
            #                 "forecast": {"product_id": item["product_id"],
            #                                 "sales_units": {k: v for k, v in zip(forecast_dates, prediction)}
            #                                 }
            #                 })
            # requests.post(get_address(URL_FORECAST), json={"data": result})

if __name__ == "__main__":
    start_time = time.time()
    _logger.info("Запуск приложения")
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    # sale = get_sales(store_id='1aa057313c28fa4a40c5bc084b11d276', product_id='00b72c2f01a1512cbb1d3f33319bac93')
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Время выполнения задачи:", execution_time, "секунд")