"""This file contains the group and group-membership models for RAG access control."""

from typing import (
    TYPE_CHECKING,
    List,
)

from sqlmodel import (
    Field,
    Relationship,
    UniqueConstraint,
)

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.document import Document

class Group(BaseModel, table=True):
    """A group that documents are scoped to for role/group-based RAG access control.

    Attributes:
        id: The primary key
        name: Unique group name
        created_at: When the group was created
        memberships: Relationship to the group's members
        documents: Relationship to documents scoped to this group
    """
    id: int = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    memberships: List["GroupMembership"] = Relationship(back_populates="group") # for python orm definition
    documents: List["Document"] = Relationship(back_populates="group") # for python orm definition

class GroupMembership(BaseModel, table=True):
    """Membership of a user in a group, with a role controlling document write access.

    Attributes:
        id: The primary key
        user_id: Foreign key to the member user
        group_id: Foreign key to the group
        role: "admin" (can add documents and manage membership) or "member" (read-only)
        created_at: When the membership was created
        group: Relationship to the owning group
    """

    __table_args__ = (UniqueConstraint("user_id", "group_id", name="uq_groupmembership_user_group"),)

    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    group_id: int = Field(foreign_key="group.id", index=True) # db only store this
    role: str = Field(default="member")
    group: "Group" = Relationship(back_populates="memberships") # for python back_populates only

# Avoid circular imports
from app.models.document import Document