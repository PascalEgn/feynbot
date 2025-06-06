"""add search feedback table

Revision ID: 849b77b6dfe6
Revises: fb7c57a4daeb
Create Date: 2025-04-02 15:31:18.956121

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "849b77b6dfe6"
down_revision: Union[str, None] = "fb7c57a4daeb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "search_feedback",
        sa.Column(
            "id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("question", sa.String(), nullable=False),
        sa.Column("additional", sa.String(), nullable=True),
        sa.Column("matomo_client_id", sa.Uuid(), nullable=True),
        sa.Column(
            "timestamp", sa.DateTime(), server_default=sa.text("now()"), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("search_feedback")
    # ### end Alembic commands ###
