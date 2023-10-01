from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTable

from sqlalchemy import Column, String, Boolean, Integer, ForeignKey


from core.db import Base


class User(SQLAlchemyBaseUserTable[int], Base):
    id = Column(Integer, primary_key=True)
    last_name = Column(String(255))
    first_name = Column(String(255))
    is_superuser = Column(Boolean)
    is_active = Column(Boolean)
    hashed_password = Column(String(255))
    email = Column(String(255), unique=True, nullable=False)
    store_id = Column(String(255), ForeignKey('store.st_id'))
