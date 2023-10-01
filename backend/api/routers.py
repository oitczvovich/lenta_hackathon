from fastapi import APIRouter

from user_auth.api.endpoints import user_router

main_router = APIRouter()


@main_router.get('/hello')
def read_root():
    return {'Hello': 'FastApi'}


main_router.include_router(user_router)
