FROM python:3.11-slim

RUN mkdir /ml_predict

COPY requirements_ml.txt /ml_predict

RUN pip3 install -r /app/requirements_ml.txt --no-cache-dir

ENV PIP_ROOT_USER_ACTION=ignore

COPY . /ml_predict

WORKDIR /ml_predict

EXPOSE 8000

# Запуск списка задач