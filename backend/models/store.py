from sqlalchemy import Column, DateTime, String, Boolean, Integer

from core.db import Base


class Store(Base):
    """Модель торговой точки."""
    st_id: str = Column(String(255), primary_key=True)
    st_city_id: str = Column(String(255))
    st_division_code: str = Column(String(255))
    st_type_format_id: int = (Column(Integer))
    st_type_loc_id: int = (Column(Integer))
    st_type_size_id: int = (Column(Integer))
    st_is_active: bool = (Column(Boolean))
