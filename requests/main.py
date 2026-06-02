import uvicorn
from fastapi import FastAPI
from router import router

app = FastAPI(
    title="Stock Data Api",
)

app.include_router(router, prefix="")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)