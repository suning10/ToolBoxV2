"""Group endpoints for RAG access control.

Groups are the access-control boundary for documents: a user can only read
documents in groups they belong to, and can only add documents to groups
where they hold the "admin" role. Membership changes here are the only way
that boundary moves.
"""

from typing import List

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)

from app.api.v1.auth import get_current_user
from app.core.config import settings
from app.core.limiter import limiter
from app.core.logging import logger
from app.models.user import User
from app.schemas.rag import (
    GroupCreate,
    GroupMemberAdd,
    GroupMemberResponse,
    GroupResponse,
)
from app.services.database import database_service
from app.services.group import group_service
from app.utils.sanitization import sanitize_string

router = APIRouter()

@router.post("/groups", response_model=GroupResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["groups"][0])
async def create_group(request: Request, group: GroupCreate, user: User = Depends(get_current_user)):
    """Create a group. The creator becomes its first admin member.

    Args:
        request: The FastAPI request object for rate limiting.
        group: The group to create.
        user: The authenticated user.

    Returns:
        GroupResponse: The created group.

    Raises:
        HTTPException: 422 if a group with this name already exists.
    """
    try:
        created = await group_service.create_group(group.name, user.id)
        logger.info("group_create_requested", group_id=created.id, user_id=user.id)
        return GroupResponse(id=created.id, name=created.name, role="admin")
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))


@router.get("/groups", response_model=List[GroupResponse])
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["groups"][0])
async def list_groups(request: Request, user: User = Depends(get_current_user)):
    """List every group the authenticated user belongs to.

    Args:
        request: The FastAPI request object for rate limiting.
        user: The authenticated user.

    Returns:
        List[GroupResponse]: Groups the user belongs to, with their role in each.
    """
    memberships = await group_service.list_user_groups(user.id)
    return [GroupResponse(id=g.id, name=g.name, role=m.role) for g, m in memberships]


@router.get("/groups/{group_id}/members", response_model=List[GroupMemberResponse])
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["groups"][0])
async def list_group_members(request: Request, group_id: int, user: User = Depends(get_current_user)):
    """List a group's members. Requires the authenticated user to be a member.

    Args:
        request: The FastAPI request object for rate limiting.
        group_id: The group to list members of.
        user: The authenticated user.

    Returns:
        List[GroupMemberResponse]: The group's members.

    Raises:
        HTTPException: 403 if the user isn't a member of this group.
    """
    if await group_service.get_membership(user.id, group_id) is None:
        raise HTTPException(status_code=403, detail="not a member of this group")

    members = await group_service.list_members(group_id)
    return [
        GroupMemberResponse(user_id=member_user.id, email=member_user.email, role=m.role) for m, member_user in members
    ]


@router.post("/groups/{group_id}/members", response_model=GroupMemberResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["groups"][0])
async def add_group_member(
    request: Request, group_id: int, member: GroupMemberAdd, user: User = Depends(get_current_user)
):
    """Add a user to a group, or update their role. Admin-only.

    Args:
        request: The FastAPI request object for rate limiting.
        group_id: The group to add a member to.
        member: The member's email and role to grant.
        user: The authenticated user — must be an admin of this group.

    Returns:
        GroupMemberResponse: The added/updated member.

    Raises:
        HTTPException: 403 if not a group admin, 404 if the target user doesn't exist.
    """
    if not await group_service.is_admin(user.id, group_id):
        raise HTTPException(status_code=403, detail="only group admins can add members")

    target_user = await database_service.get_user_by_email(sanitize_string(member.email))
    if target_user is None:
        raise HTTPException(status_code=404, detail="user not found")

    membership = await group_service.add_member(group_id, target_user.id, member.role)
    logger.info("group_member_add_requested", group_id=group_id, admin_user_id=user.id, target_user_id=target_user.id)
    return GroupMemberResponse(user_id=target_user.id, email=target_user.email, role=membership.role)

@router.delete("/groups/{group_id}/members/{target_user_id}")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["groups"][0])
async def remove_group_member(
    request: Request, group_id: int, target_user_id: int, user: User = Depends(get_current_user)
):
    """Remove a user from a group. Admin-only.

    Args:
        request: The FastAPI request object for rate limiting.
        group_id: The group to remove a member from.
        target_user_id: The member to remove.
        user: The authenticated user — must be an admin of this group.

    Returns:
        dict: A confirmation message.

    Raises:
        HTTPException: 403 if not a group admin, 404 if the target wasn't a member.
    """
    if not await group_service.is_admin(user.id, group_id):
        raise HTTPException(status_code=403, detail="only group admins can remove members")

    removed = await group_service.remove_member(group_id, target_user_id)
    if not removed:
        raise HTTPException(status_code=404, detail="user is not a member of this group")

    logger.info(
        "group_member_remove_requested", group_id=group_id, admin_user_id=user.id, target_user_id=target_user_id
    )
    return {"message": "Member removed successfully"}