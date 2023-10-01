from fastapi import FastAPI

from user_auth.api.routers import main_router
from user_auth.core.config import settings
from user_auth.core.init_db import create_first_superuser

app = FastAPI(title=settings.app_title)

app.include_router(main_router)


@app.on_event('startup')
async def startup():
    await create_first_superuser()
