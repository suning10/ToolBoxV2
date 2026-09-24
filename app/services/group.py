"""This file contains the group service for RAG access control."""

from typing import (
    List,
    Optional,
    Tuple,
)

from sqlmodel import (
    Session,
    col,
    select,
)

from app.core.logging import logger
from app.models import session
from app.models.group import (
    Group,
    GroupMembership,
)
from app.models.user import User
from app.services.database import database_service

class GroupService:
    """Service for managing groups and memberships that gate RAG document access."""

    def __init__(self):
        """Reuse the shared database engine rather than opening a second pool."""
        self.engine = database_service.engine

    async def create_group(self, name: str, creator_user_id: int) -> Group:
        """Create a group and make its creator an admin member.

        Args:
            name: The unique group name.
            creator_user_id: The user who becomes the group's first admin.

        Returns:
            Group: The created group.

        Raises:
            ValueError: If a group with this name already exists.
        """
        with Session(self.engine) as session:
            if session.exec(select(Group).where(Group.name == name)).first():
                raise ValueError(f"Group {name} already exists.")

            group = Group(name=name)
            session.add(group)
            session.commit()
            session.refresh(group)

            membership = GroupMembership(user_id=creator_user_id, group_id=group.id, role="admin")
            session.add(membership)
            session.commit()

            logger.info("group_created", group_id=group.id, name=name, creator_user_id=creator_user_id)
            return group

    async def get_group(self, group_id: int) -> Optional[Group]:
        """Get a group by ID."""
        with Session(self.engine) as session:
            return session.get(Group, group_id)

    async def get_membership(self, user_id: int, group_id: int) -> Optional[GroupMembership]:
        """Get a user's membership record for a group, or None if not a member."""
        with Session(self.engine) as session:
            statement = select(GroupMembership).where(
                col(GroupMembership.user_id) == user_id,
                col(GroupMembership.group_id) == group_id
            )
        return session.exec(statement).first()

    async def is_admin(self, user_id: int, group_id: int) -> bool:
        """Return whether the user is an admin member of the group."""
        membership = await self.get_membership(user_id, group_id)
        return membership is not None and membership.role == "admin"

    async def add_member(self, group_id: int, target_user_id: int, role: str = "member") -> GroupMembership:
        """Add a user to a group, or update their role if already a member.

        Args:
            group_id: The group to add the user to.
            target_user_id: The user being added.
            role: "admin" or "member".

        Returns:
            GroupMembership: The created or updated membership.
        """
        with Session(self.engine) as session:
            membership = GroupMembership(user_id=target_user_id, group_id=group_id, role=role)
            session.add(membership)
            session.commit()
            session.refresh(membership)
            logger.info("group_member_added", group_id=group_id, user_id=target_user_id, role=role)
            return membership

    async def remove_member(self, group_id: int, target_user_id: int) -> bool:
        """Remove a user from a group.

        Returns:
            bool: True if a membership was removed, False if the user wasn't a member.
        """
        with Session(self.engine) as session:
            statement = select(GroupMembership).where(col(GroupMembership.group_id) == group_id,
                                                      col(GroupMembership.user_id) == target_user_id)
            membership = session.exec(statement).first()
            if not membership:
                return False
            session.delete(membership)
            session.commit()
            logger.info("group_member_removed", group_id=group_id, user_id=target_user_id)
            return True

    async def list_user_groups(self, user_id: int) -> List[Tuple[Group, GroupMembership]]:
        """List every group a user belongs to, with their role in each."""
        with Session(self.engine) as session:
            statement = (
                select(Group, GroupMembership)
                .join(
                    GroupMembership, col(GroupMembership.group_id) == col(Group.id),
                )
                .where(col(GroupMembership.user_id) == user_id)
            )
            return list(session.exec(statement).all())

    async def list_members(self, group_id: int) -> List[Tuple[GroupMembership, User]]:
        """List every member of a group, with their user record."""
        with Session(self.engine) as session:
            statement = (
                select(GroupMembership, User)
                .join(User, col(User.id == GroupMembership.user_id))
                .where(GroupMembership.group_id == group_id)
            )

            return list(session.exec(statement).all())

    async def user_group_ids(self, user_id: int) -> List[int]:
        """Return the IDs of every group a user belongs to — the RAG access boundary."""
        with Session(self.engine) as session:
            statement = select(GroupMembership.group_id).where(col(GroupMembership.user_id) == user_id)
            return list(session.exec(statement).all())

group_service = GroupService()


