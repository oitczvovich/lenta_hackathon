import pandas as pd

class HybridModel:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None  
        
def fit(self, X_1, X_2, y):
    # Обучаем первую модель
    self.model_1.fit(X_1, y)

    # Делаем предсказания
    y_fit = pd.DataFrame(
        self.model_1.predict(X_1), 
        index=X_1.index, columns=y.columns,
    )

    # Считаем остатки
    y_resid = y - y_fit
    y_resid = y_resid.stack().squeeze() 
# Обучаем вторую модель на остатках
    self.model_2.fit(X_2, y_resid)

    # Сохраняем данные 
    self.y_columns = y.columns
    
    self.y_fit = y_fit
    self.y_resid = y_resid


def predict(self, X_1, X_2):
    # Предсказываем первой моделью
    y_pred_1 = pd.DataFrame(
        self.model_1.predict(X_1), 
        index=X_1.index, columns=self.y_columns,
    )
    
    # Предсказываем второй моделью остатки
    y_pred_2 = pd.DataFrame(
        self.model_2.predict(X_2),
        index=X_2.index, columns=self.y_columns,
    )
    
    # Добавляем предсказания модели 2 к предсказаниям модели 1
    y_pred = y_pred_1 + y_pred_2
    
    return y_pred


HybridModel.fit = fit
HybridModel.predict = predict