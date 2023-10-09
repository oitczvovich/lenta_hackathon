import requests
import os
import logging
from datetime import date, timedelta, datetime
import json
import pandas as pd
import csv
import time
from dotenv import load_dotenv

from core.settings import settings
from core.logger import _logger


URL_CATEGORIES = "categories"
URL_SALES = "sales"
URL_STORES = "shops"
URL_FORECAST = "forecast"
URL_LOGIN = "login"
URL_REFRESH = "refresh"

NOW_DATE=date.today()
PREDICTION_DAYS: int = 14


api_port = settings.api_port
api_host = settings.api_host
name_user = settings.name_user
password_user = settings.password_user



class WorkWithAPI:

    def __init__(self):
        self.token_access = None
        self.token_refresh = None
        self.token_expiration = None
        self.headers = {"Authorization": f"Bearer {self.token_access}"}


    def handle_token(self):
        if not self.token_refresh:
            # First time accessing the class, send user_data to get the initial token
            url_login = self.get_address(URL_LOGIN)
            user_data = {"username": name_user, "password": password_user}
            r = requests.post(url_login, data=user_data)

            if r.status_code == 200:
                token = r.json().get('access')
                expiration = datetime.now() + timedelta(days=1)
                self.token = token
                self.token_expiration = expiration
                _logger.info('Token получен')
            else:
                _logger.warning('Failed to get initial token')
                return

        # Use token_refresh to refresh the access token
        url_refresh = self.get_address(URL_REFRESH)
        refresh_data = {"refresh": self.token_refresh}
        r = requests.post(url_refresh, data=refresh_data)

        if r.status_code == 200:
            token = r.json().get('access')
            expiration = datetime.now() + timedelta(days=1)
            self.token = token
            self.token_expiration = expiration
            _logger.info('Token обновлен')
        else:
            _logger.warning('Failed to refresh token')


    def get_address(self, resource):
        return "http://" + api_host + ":" + api_port + "/" + resource
    
    # функции запросы к БД.
    
    def get_stores(self):
        """ Запрос для получения списка магазинов.
        """
        stores_url = self.get_address(URL_STORES)
        resp = requests.get(stores_url, headers=self.headers)
        if resp.status_code != 200:
            _logger.warning("Could not get stores list")
            return [], resp.status_code
        return resp.json()["data"]
        
    def get_sales(self, store_id=None, product_id=None):
        """
        """
        sale_url = self.get_address(URL_SALES)
        params = {}
        if store_id is not None:
            params["store_id"] = store_id
        if product_id is not None:
            params["product_id"] = product_id
        resp = requests.get(sale_url, params=params, headers=self.headers)
        if resp.status_code != 200:
            _logger.warning("Could not get sales history")
            return []
        
        return resp.json()["data"]


    def get_categs_info(self):
        """

        """
        categs_url = self.get_address(URL_CATEGORIES)
        resp = requests.get(categs_url, headers=self.headers)
        if resp.status_code != 200:
            _logger.warning("Could not get category info")
            return {}
        result = {el["pr_sku_id"]: el for el in resp.json()["data"]}
        return result
