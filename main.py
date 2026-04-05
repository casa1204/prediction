"""FastAPI 앱 진입점 - XRP 가격 예측 대시보드."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import router
from db.database import init_db
from scheduler.job_scheduler import create_scheduler

logger = logging.getLogger(__name__)

# 프론트엔드 빌드 디렉토리
_FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan 이벤트: 시작 시 DB 초기화 + 스케줄러 시작, 종료 시 스케줄러 정지."""
    # 시작
    init_db()
    logger.info("DB 초기화 완료")

    scheduler = create_scheduler()
    scheduler.start()
    logger.info("스케줄러 시작")

    yield

    # 종료
    scheduler.shutdown(wait=False)
    logger.info("스케줄러 종료")


app = FastAPI(
    title="XRP Price Prediction Dashboard",
    description="XRP 가격 예측 대시보드 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# 프론트엔드 정적 파일 서빙 (프로덕션)
if _FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """SPA 라우팅: API가 아닌 모든 경로를 index.html로."""
        file_path = _FRONTEND_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    async def root():
        """헬스 체크 엔드포인트."""
        return {"status": "ok", "message": "XRP Price Prediction Dashboard API"}
