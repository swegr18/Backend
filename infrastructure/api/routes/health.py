"""health check routes"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """health check"""
    return {"status": "healthy"}


@router.get("/")
async def root():
    """health check"""
    return {"message": "Welcome to the API"}
