from fastapi import FastAPI

from api import router as translate_router


def create_app() -> FastAPI:
    app = FastAPI(title="LM Studio Translator API")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    app.include_router(translate_router)
    return app


app = create_app()

