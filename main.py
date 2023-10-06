import requests
import os
import logging
from datetime import date, timedelta, datetime
import json
import pandas as pd
import csv
import time


from model import forecast

current_direct = os.path.join(os.getcwd(), 'out_data') 

NOW_DATE=date.today()
PREDICTION_DAYS: int = 14


URL_CATEGORIES = "categories"
URL_SALES = "sales"
URL_STORES = "shops"
URL_FORECAST = "forecast"

FORMAT_DATE = '%d.%m.%Y'


api_port = os.environ.get("API_PORT", "8000/api/v1")
api_host = os.environ.get("API_PORT", "127.0.0.1")

_logger = logging.getLogger(__name__)


# predict = os.path.join(os.getcwd(), 'predict.csv')


def setup_logging():
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.DEBUG)
    handler_m = logging.StreamHandler()
    formatter_m = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
    handler_m.setFormatter(formatter_m)
    _logger.addHandler(handler_m)


def get_address(resource):
    return "http://" + api_host + ":" + api_port + "/" + resource




# функции запросы к БД.
def get_stores():
    """ Запрос для получения списка магазинов.
    """
    stores_url = get_address(URL_STORES)
    # print(stores_url)
    resp = requests.get(stores_url, headers=headers)
    if resp.status_code != 200:
        _logger.warning("Could not get stores list")
        return [], resp.status_code
    return resp.json()["data"]
    
def get_sales(store_id=None, product_id=None):
    """
     "data": [
        {
            "store_id": "c81e728d9d4c2f636f067f89cc14862c",
            "product_id": "fe50ae64d08d4f8245aaabc55d1baf79",
            "fact": [
                {
                    "date": "2022-08-31",
                    "pr_sales_type_id": true,
                    "pr_sales_in_units": 0,
                    "pr_promo_sales_in_units": 0,
                    "pr_sales_in_rub": 61.0,
                    "pr_promo_sales_in_rub": 61.0
                },
                {
                    "date": "2023-03-14",
                    "pr_sales_type_id": false,
                    "pr_sales_in_units": 4,
                    "pr_promo_sales_in_units": 0,
                    "pr_sales_in_rub": 564.0,
                    "pr_promo_sales_in_rub": 0.0
                },
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
    {"fd064933250b0bfe4f926b867b0a5ec8":
        {
            "pr_sku_id": "fd064933250b0bfe4f926b867b0a5ec8",
            "pr_group_id": "c74d97b01eae257e44aa9d5bade97baf",
            "pr_cat_id": "1bc0249a6412ef49b07fe6f62e6dc8de",
            "pr_subcat_id": "ca34f669ae367c87f0e75dcae0f61ee5",
            "pr_uom_id": 17
        },
    }
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
    #TODO переписать на .env
    url_login = 'http://127.0.0.1:8000/api/v1/login'
    user_data = {"username": "SuperUser", "password": "GERvre4tvaSAAG453gr"}
        
    r = requests.post(url_login, data=user_data)
    
    if r.status_code == 200:
        token = r.json().get('access')
        return token
    else:
        return None



def write_json(val, name):
    with open(f"{name}.json", "w") as f:
        json.dump(val, f)

def craet_csv(data_dict: dict, name_file: str):
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


def create_subm_data(item_info, store):
    """ Подготовка данных для предсказания.
    st_id – захэшированное id магазина;
    pr_sku_id – захэшированное id товара;
    date – дата (день) текущий день + 14 дней.
    Возвращается DataFrame.
    """
    DF_subm_data = pd.DataFrame()
    forecast_dates = [NOW_DATE + timedelta(days=d) for d in range(1, PREDICTION_DAYS + 1)]
    forecast_dates = [el.strftime('%Y-%m-%d') for el in forecast_dates]

    for forecast_day in forecast_dates:
        print('forecast_day', forecast_day, type(forecast_day))
        
        data_dict = {
            'st_id': store['st_id'],
            'pr_sku_id': item_info['pr_sku_id'],
            'date': forecast_day,
            'target': 0,
        }
        craet_csv(data_dict, 'subm_data')
        DF_subm_data = DF_subm_data._append(data_dict, ignore_index=True)
        DF_subm_data['date'] = pd.to_datetime(DF_subm_data['date'], format='%Y-%m-%d')
    return DF_subm_data


def create_holidays():
    """ Создание Dataframe из файла holidays_covid_calendar_2
    Обработка только поля date, т.к. другие данные не нужны.
    """
    holidays_path = os.path.join(os.getcwd(), 'holidays_covid_calendar_2.csv')
    holidays_df = pd.read_csv(holidays_path)
    data_dict = {
        'year': holidays_df['year'],
        'day': holidays_df['day'],
        'weekday': holidays_df['weekday'],
        'date': holidays_df['date'],
        'calday': holidays_df['calday'],
        'holiday': holidays_df['holiday'],
        'weekend': holidays_df['weekend'],
        'truly_holidays': holidays_df['truly_holidays'],
    }
    DF_holidays = pd.DataFrame()
    DF_holidays = DF_holidays._append(data_dict, ignore_index=True)
    # DF_holidays.info()

    return DF_holidays



def create_train_data(sales, item_info, store):
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
1   1aa057313c28fa4a40c5bc084b11d276  fd064933250b0bfe4f926b867b0a5ec8               3  2022-08-08                 0                  3             73.0
    """
    
    # здесь пытался сделать описание колонок, но появлялась ошибка пока все закоментил.
    DF_train_data = pd.DataFrame(
        # columns=[
        #     'st_id',
        #     'pr_sku_id',
        #     'pr_group_id',
        #     'pr_cat_id',
        #     'pr_subcat_id',
        #     'pr_uom_id',
        #     'st_is_active',
        #     'st_city_id',
        #     'st_division_code',
        #     'st_type_format_id',
        #     'st_type_loc_id',
        #     'st_type_size_id',
        #     'date',
        #     'pr_sales_type_id',
        #     'pr_sales_in_units',
        #     'pr_sales_in_rub',
        #     ],
        # dtype={
        #     'st_id': str,
        #     'pr_sku_id': str,
        #     'pr_group_id': str,
        #     'pr_cat_id': str,
        #     'pr_subcat_id': str,
        #     'pr_uom_id': str,
        #     'st_is_active': str,
        #     'st_city_id': str,
        #     'st_division_code': str,
        #     'st_type_format_id': str,
        #     'st_type_loc_id': str,
        #     'st_type_size_id': str,
        #     'date': 'datetime64[ns]',
        #     'pr_sales_type_id': str,
        #     'pr_sales_in_units': float,
        #     'pr_sales_in_rub': float,
        #     }
        )

    st_pr_data_dict = {
        'st_id': store['st_id'],
        'pr_sku_id': item_info['pr_sku_id'],
        'pr_group_id': item_info['pr_group_id'],
        'pr_cat_id': item_info['pr_cat_id'],
        'pr_subcat_id': item_info['pr_subcat_id'],
        'pr_uom_id': str(item_info['pr_uom_id']),
        'st_is_active': store['st_is_active'],
        'st_city_id': store['st_city_id'],
        'st_division_code': store['st_division_code'],
        'st_type_format_id': store['st_type_format_id'],
        'st_type_loc_id': str(store['st_type_loc_id']),
        'st_type_size_id': store['st_type_size_id'],
    }

    for sales_pr in sales:
        data_dict = {
            'date':  sales_pr['date'],
            'pr_sales_type_id': sales_pr['pr_sales_type_id'],
            'pr_sales_in_units': float(sales_pr['pr_sales_in_units']),
            'pr_sales_in_rub': sales_pr['pr_sales_in_rub'],
        }
        
        st_pr_data_dict.update(data_dict)
        DF_train_data = DF_train_data._append(st_pr_data_dict, ignore_index=True)
        DF_train_data['date'] = pd.to_datetime(DF_train_data['date'], format='%Y-%m-%d')
        DF_train_data.info()
    return DF_train_data





def main(today=date.today()):
    forecast_dates = [today + timedelta(days=d) for d in range(1, 6)]
    forecast_dates = [el.strftime('%d.%m.%Y') for el in forecast_dates]

    categs_info = get_categs_info()  # список product_id
    for store in get_stores():
        result = []
        predict = []
        store_id = store["st_id"]
        for product_id in get_categs_info():
            # получаем id продукта
            for item in get_sales(store_id=store_id, product_id=product_id):
                item_info = categs_info[item["product_id"]]
                sales = item["fact"]
                # создаем DataFrame для передачи в forecast
                subm_data = create_subm_data(item_info, store)
                train_data = create_train_data(sales, item_info, store)
                holidays = create_holidays()
                forecast(subm_data, train_data, holidays)
                # функция forecsct возвращает dataFraeme со следующими колонками.
        #                 data_dict = {
                #     'st_id': store['st_id'],
                #     'pr_sku_id': item_info['pr_sku_id'],
                #     'date': forecast_day,
                #     'target': 0,
                # }
                
                # здесь обработка результатов. 
                
                
                
            #     prediction = forecast(sales, item_info, store)
                
            #     result.append({"store": store["store"],
            #                 "forecast_date": today.strftime("%Y-%m-%d"),
            #                 "forecast": {"product_id": item["product_id"],
            #                                 "sales_units": {k: v for k, v in zip(forecast_dates, prediction)}
            #                                 }
            #                 })
            # requests.post(get_address(URL_FORECAST), json={"data": result})

if __name__ == "__main__":
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    setup_logging()
    start_time = time.time()

    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Время выполнения задачи:", execution_time, "секунд")