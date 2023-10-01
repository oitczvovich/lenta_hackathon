"""First migration

Revision ID: dd032b921af0
Revises: 
Create Date: 2023-10-01 10:10:37.513539

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dd032b921af0'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('product',
    sa.Column('pr_sku_id', sa.String(length=255), nullable=False),
    sa.Column('pr_group_id', sa.String(length=255), nullable=True),
    sa.Column('pr_cat_id', sa.String(length=255), nullable=True),
    sa.Column('pr_subcat_id', sa.String(length=255), nullable=True),
    sa.Column('pr_uom_id', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('pr_sku_id')
    )
    op.create_table('store',
    sa.Column('st_id', sa.String(length=255), nullable=False),
    sa.Column('st_city_id', sa.String(length=255), nullable=True),
    sa.Column('st_division_code', sa.String(length=255), nullable=True),
    sa.Column('st_type_format_id', sa.Integer(), nullable=True),
    sa.Column('st_type_loc_id', sa.Integer(), nullable=True),
    sa.Column('st_type_size_id', sa.Integer(), nullable=True),
    sa.Column('st_is_active', sa.Boolean(), nullable=True),
    sa.PrimaryKeyConstraint('st_id')
    )
    op.create_table('salesedu',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('st_id', sa.String(length=255), nullable=True),
    sa.Column('pr_sku_id', sa.String(length=255), nullable=True),
    sa.Column('date', sa.DateTime(), nullable=True),
    sa.Column('pr_sales_type_id', sa.Boolean(), nullable=True),
    sa.Column('pr_sales_in_units', sa.Integer(), nullable=True),
    sa.Column('pr_promo_sales_in_units', sa.Integer(), nullable=True),
    sa.Column('pr_sales_in_rub', sa.Float(), nullable=True),
    sa.Column('pr_promo_sales_in_rub', sa.Float(), nullable=True),
    sa.ForeignKeyConstraint(['pr_sku_id'], ['product.pr_sku_id'], ),
    sa.ForeignKeyConstraint(['st_id'], ['store.st_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('salesforecast',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('st_id', sa.String(length=255), nullable=True),
    sa.Column('pr_sku_id', sa.String(length=255), nullable=True),
    sa.Column('date', sa.DateTime(), nullable=True),
    sa.Column('target', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['pr_sku_id'], ['product.pr_sku_id'], ),
    sa.ForeignKeyConstraint(['st_id'], ['store.st_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('user',
    sa.Column('is_verified', sa.Boolean(), nullable=False),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('last_name', sa.String(length=255), nullable=True),
    sa.Column('first_name', sa.String(length=255), nullable=True),
    sa.Column('is_superuser', sa.Boolean(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('hashed_password', sa.String(length=255), nullable=True),
    sa.Column('email', sa.String(length=255), nullable=False),
    sa.Column('store_id', sa.String(length=255), nullable=True),
    sa.ForeignKeyConstraint(['store_id'], ['store.st_id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('user')
    op.drop_table('salesforecast')
    op.drop_table('salesedu')
    op.drop_table('store')
    op.drop_table('product')
    # ### end Alembic commands ###