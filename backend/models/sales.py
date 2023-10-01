from sqlalchemy import (
    Column,
    String,
    Boolean,
    Integer,
    DateTime,
    Float,
    ForeignKey
)
from core.db import Base


class SalesEdu(Base):
    """Модель продаж скользящая для обучения."""
    id = Column(Integer, primary_key=True)
    st_id: str = Column(String(255), ForeignKey('store.st_id'))
    pr_sku_id: str = Column(String(255), ForeignKey('product.pr_sku_id'))
    date: str = Column(DateTime)
    pr_sales_type_id: bool = (Column(Boolean))
    pr_sales_in_units: int = (Column(Integer))
    pr_promo_sales_in_units: int = (Column(Integer))
    pr_sales_in_rub: float = (Column(Float))
    pr_promo_sales_in_rub: float = (Column(Float))


class SalesForecast(Base):
    """Модель продаж предсказание."""
    id = Column(Integer, primary_key=True)
    st_id: str = Column(String(255), ForeignKey('store.st_id'))
    pr_sku_id: str = Column(String(255), ForeignKey('product.pr_sku_id'))
    date: str = Column(DateTime)
    target: int = (Column(Integer))