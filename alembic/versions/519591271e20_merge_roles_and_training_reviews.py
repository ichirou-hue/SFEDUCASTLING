"""merge roles and training reviews

Revision ID: 519591271e20
Revises: a1b2c3d4e5f6, e4a51c7d92f1
Create Date: 2026-10-04 19:45:42.586895
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '519591271e20'
down_revision: Union[str, None] = ('a1b2c3d4e5f6', 'e4a51c7d92f1')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass