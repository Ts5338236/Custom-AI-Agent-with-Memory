from fastapi import APIRouter, Depends
from app.api.deps import RoleChecker
from app.models.chat import User

router = APIRouter()

# Only accessible by users with 'admin' role
allow_admin = RoleChecker(["admin"])

@router.get("/stats", dependencies=[Depends(allow_admin)])
async def get_system_stats():
    """
    Returns high-level system statistics (Admin only).
    """
    return {
        "active_users": 10,
        "total_messages": 1000,
        "vector_db_size": "15MB"
    }
