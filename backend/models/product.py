from sqlalchemy import Column, String, Integer

from core.db import Base


class Product(Base):
    """Модель товар."""
    pr_sku_id: str = Column(String(255), primary_key=True)
    pr_group_id: str = Column(String(255))
    pr_cat_id: str = Column(String(255))
    pr_subcat_id: str = Column(String(255))
    pr_uom_id: int = (Column(Integer))
