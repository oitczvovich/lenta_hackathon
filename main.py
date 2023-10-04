import os
from datetime import date
import pandas as pd
import time

from model import forecast

def main(today=date.today()):
    holidays = os.path.join(os.getcwd(), 'holidays_covid_calendar_2.csv')
    subm_data = pd.read_csv('subm_data.csv')
    train_data = pd.read_csv('predict.csv')
    forecast(subm_data, train_data, holidays)
                

if __name__ == "__main__":
    start_time = time.time()

    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Время выполнения задачи:", execution_time, "секунд")