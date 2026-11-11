"""
PruneQuest Studio API routes package.
"""

from fastapi import APIRouter
from prunequest.studio.api.routes_system import router as system_router
from prunequest.studio.api.routes_models import router as models_router
from prunequest.studio.api.routes_benchmarks import router as benchmarks_router
from prunequest.studio.api.routes_optimization import router as optimization_router
from prunequest.studio.api.routes_runs import router as runs_router

api_router = APIRouter()
api_router.include_router(system_router)
api_router.include_router(models_router)
api_router.include_router(benchmarks_router)
api_router.include_router(optimization_router)
api_router.include_router(runs_router)

__all__ = ["api_router"]
