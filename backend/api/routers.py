from fastapi import APIRouter

from api.endpoints import user_router


main_router = APIRouter()


@main_router.get('/ping')
def read_root():
    return {'Hello': 'FastApi'}


main_router.include_router(user_router)
