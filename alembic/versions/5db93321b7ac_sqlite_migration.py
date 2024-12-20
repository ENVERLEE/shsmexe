"""sqlite_migration

Revision ID: 5db93321b7ac
Revises: 
Create Date: 2024-11-17 13:31:54.975344

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5db93321b7ac'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('email', sa.String(length=100), nullable=False),
    sa.Column('hashed_password', sa.String(length=200), nullable=False),
    sa.Column('full_name', sa.String(length=100), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email')
    )
    op.create_table('projects',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('title', sa.String(length=200), nullable=False),
    sa.Column('description', sa.String(length=1000), nullable=True),
    sa.Column('evaluation_plan', sa.String(length=1000), nullable=True),
    sa.Column('submission_format', sa.String(length=500), nullable=True),
    sa.Column('metadata1', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('research_field', sa.String(length=100), nullable=True),
    sa.Column('evaluation_status', sa.String(length=50), nullable=True),
    sa.Column('final_score', sa.Numeric(precision=5, scale=2), nullable=True),
    sa.Column('submission_date', sa.DateTime(), nullable=True),
    sa.Column('total_steps', sa.Integer(), nullable=True),
    sa.Column('completed_steps', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('reference_materials',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('project_id', sa.Integer(), nullable=True),
    sa.Column('title', sa.String(length=200), nullable=True),
    sa.Column('authors', sa.JSON(), nullable=True),
    sa.Column('publication_date', sa.DateTime(), nullable=True),
    sa.Column('content', sa.String(length=2000), nullable=True),
    sa.Column('metadata1', sa.JSON(), nullable=True),
    sa.Column('embedding', sa.JSON(), nullable=True),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('research_steps',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('project_id', sa.Integer(), nullable=True),
    sa.Column('step_number', sa.Integer(), nullable=True),
    sa.Column('description', sa.String(length=1000), nullable=True),
    sa.Column('keywords', sa.JSON(), nullable=True),
    sa.Column('methodology', sa.String(length=500), nullable=True),
    sa.Column('output_format', sa.String(length=200), nullable=True),
    sa.Column('status', sa.String(length=50), nullable=True),
    sa.Column('result', sa.JSON(), nullable=True),
    sa.Column('executed_at', sa.DateTime(), nullable=True),
    sa.Column('started_at', sa.DateTime(), nullable=True),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.Column('error_message', sa.String(length=500), nullable=True),
    sa.Column('progress_percentage', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('research_steps')
    op.drop_table('reference_materials')
    op.drop_table('projects')
    op.drop_table('users')
    # ### end Alembic commands ###
