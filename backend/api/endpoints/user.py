from http import HTTPStatus

from fastapi import APIRouter, HTTPException

from user_auth.core.user import auth_backend, fastapi_users
from user_auth.shemas.user import UserCreate, UserRead, UserUpdate

router = APIRouter()

router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix='/auth/jwt',
    tags=['Auth']
)

router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix='/auth',
    tags=['Auth']
)

router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix='/users',
    tags=['Users']
)


@router.delete(
    '/users/{id}',
    tags=['Users'],
    deprecated=True
)
def delete_user(id: str):
    """Don't use delete, deactivate users."""
    raise HTTPException(
        status_code=HTTPStatus.METHOD_NOT_ALLOWED,
        detail="Deleting users is forbidden!"
    )