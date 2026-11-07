from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import case, func, inspect, or_, select
from sqlalchemy.exc import SQLAlchemyError

from src.database.connection import get_engine
from src.database.models import (
    Base, Category, DailySale, Inventory, Order, OrderItem, Product,
    Supplier, SupplierDelivery, Warehouse,
)
from src.analytics.customer_segmentation import build_customer_segment_report
from src.analytics.metrics import days_inventory_outstanding, inventory_holding_cost, xyz_classification
from src.forecasting.pipeline import forecast_product
from src.recommendations.engine import generate_recommendations

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("smart_inventory.api")


def _read(request: Request, statement) -> list[dict]:
    try:
        with request.app.state.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings().all()]
    except SQLAlchemyError as exc:
        logger.exception("Database query failed")
        raise HTTPException(status_code=503, detail="Database query failed; check the configured database and schema.") from exc


def _frame(request: Request, statement) -> pd.DataFrame:
    return pd.DataFrame(_read(request, statement))


def _scalar(request: Request, statement):
    try:
        with request.app.state.engine.connect() as connection:
            return connection.execute(statement).scalar_one_or_none()
    except SQLAlchemyError as exc:
        logger.exception("Database scalar query failed")
        raise HTTPException(status_code=503, detail="Database query failed; check the configured database and schema.") from exc


def _ensure_schema_compatibility(engine) -> None:
    try:
        with engine.begin() as connection:
            inspector = inspect(connection)
            if "orders" in inspector.get_table_names():
                order_columns = {column["name"] for column in inspector.get_columns("orders")}
                if "customer_id" not in order_columns:
                    connection.exec_driver_sql("ALTER TABLE orders ADD COLUMN customer_id INTEGER")
    except SQLAlchemyError:
        logger.exception("Unable to apply schema compatibility migration for customer_id")
        raise


def _stock_frame(request: Request) -> pd.DataFrame:
    statement = select(
        Inventory.product_id,
        Product.description.label("product_name"),
        Product.supplier_id,
        func.sum(Inventory.quantity_on_hand).label("quantity_on_hand"),
        func.sum(Inventory.quantity_on_hand * Inventory.estimated_unit_cost).label("inventory_value"),
        func.min(Inventory.data_source).label("data_source"),
    ).join(Product, Product.product_id == Inventory.product_id).group_by(
        Inventory.product_id, Product.description, Product.supplier_id
    )
    return _frame(request, statement)


def _demand_by_product(request: Request, start_date: date | None = None) -> pd.DataFrame:
    statement = select(
        DailySale.product_id,
        func.sum(DailySale.units_sold).label("units_sold"),
    )
    if start_date:
        statement = statement.where(DailySale.sales_date >= start_date)
    return _frame(request, statement.group_by(DailySale.product_id))


def _daily_demand(request: Request, start_date: date | None = None) -> pd.DataFrame:
    statement = select(
        DailySale.product_id,
        DailySale.sales_date,
        func.sum(DailySale.units_sold).label("units_sold"),
    )
    if start_date:
        statement = statement.where(DailySale.sales_date >= start_date)
    return _frame(request, statement.group_by(DailySale.product_id, DailySale.sales_date))


def create_app(database_url: str | None = None) -> FastAPI:
    engine = get_engine(database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            Base.metadata.create_all(engine)
            _ensure_schema_compatibility(engine)
            app.state.engine = engine
            logger.info("Database schema is ready")
        except SQLAlchemyError:
            logger.exception("Unable to initialize database schema")
            raise
        yield
        engine.dispose()

    app = FastAPI(
        title="Smart Inventory & Supply Chain Analytics",
        description=(
            "Retail analytics built from UCI Online Retail II. Inventory, supplier, warehouse, "
            "cost, and inferred category fields are explicitly synthetic demonstrations."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.engine = engine
    origins = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in origins],
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health(request: Request):
        try:
            engine = getattr(request.app.state, "engine", None) or app.state.engine
            with engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
            return {"status": "ok", "database": "connected"}
        except SQLAlchemyError as exc:
            logger.exception("Database health check failed")
            raise HTTPException(status_code=503, detail="Database is unavailable.") from exc

    @app.get("/api/dashboard/overview")
    def overview(request: Request, start_date: date | None = None, end_date: date | None = None):
        item_query = select(
            func.sum(OrderItem.line_revenue).label("revenue"),
            func.sum(OrderItem.estimated_cogs).label("cogs"),
            func.sum(OrderItem.quantity).label("units_sold"),
        ).select_from(OrderItem)
        if start_date:
            item_query = item_query.join(Order, Order.order_id == OrderItem.order_id).where(Order.order_date >= start_date)
        if end_date:
            if not start_date:
                item_query = item_query.join(Order, Order.order_id == OrderItem.order_id)
            item_query = item_query.where(Order.order_date < pd.Timestamp(end_date) + pd.Timedelta(days=1))
        item_values = _read(request, item_query)[0]
        order_query = select(func.count(func.distinct(Order.order_id)))
        if start_date:
            order_query = order_query.where(Order.order_date >= start_date)
        if end_date:
            order_query = order_query.where(Order.order_date < pd.Timestamp(end_date) + pd.Timedelta(days=1))
        orders_count = int(_scalar(request, order_query) or 0)
        products_count = int(_scalar(request, select(func.count(Product.product_id)) ) or 0)
        categories_count = int(_scalar(request, select(func.count(Category.category_id)) ) or 0)
        revenue = float(item_values["revenue"] or 0)
        cogs = float(item_values["cogs"] or 0)
        units_sold = float(item_values["units_sold"] or 0)
        snapshot_inventory_value = float(_scalar(
            request,
            select(func.sum(Inventory.quantity_on_hand * Inventory.estimated_unit_cost)),
        ) or 0)
        inventory_snapshot_rows = _read(
            request,
            select(
                (Inventory.quantity_on_hand * Inventory.estimated_unit_cost).label("inventory_value")
            )
            .where(Inventory.as_of_date.is_not(None)),
        )
        average_inventory_value_estimated = (
            float(pd.DataFrame(inventory_snapshot_rows)["inventory_value"].mean())
            if inventory_snapshot_rows else snapshot_inventory_value
        )
        zero_stock_count = int(_scalar(
            request,
            select(func.sum(case((Inventory.quantity_on_hand == 0, 1), else_=0))),
        ) or 0)
        inventory_rows = int(_scalar(request, select(func.count()).select_from(Inventory)) or 0)
        latest = _scalar(request, select(func.max(DailySale.sales_date)))
        overstock_value = 0.0
        if latest is not None:
            cutoff = pd.Timestamp(latest).date() - pd.Timedelta(days=89)
            demand = _demand_by_product(request, cutoff)
            stock = _stock_frame(request)
            if not demand.empty and not stock.empty:
                rate = demand.set_index("product_id").units_sold.div(90)
                stock["daily_demand"] = stock.product_id.map(rate).fillna(0)
                coverage = stock.quantity_on_hand.div(stock.daily_demand.replace(0, pd.NA))
                stock["coverage"] = coverage
                overstock_value = float(stock.loc[stock.coverage > (
                    int(os.getenv("TARGET_COVERAGE_DAYS", "45"))
                    * float(os.getenv("RECOMMENDATION_STOCK_MULTIPLIER", "1.5"))
                ), "inventory_value"].sum())
        return {
            "revenue": revenue,
            "orders": int(orders_count),
            "products": int(products_count),
            "categories": int(categories_count),
            "inventory_value_snapshot_estimated": snapshot_inventory_value,
            "average_inventory_value_estimated": average_inventory_value_estimated,
            "inventory_turnover_estimated": (
                cogs / average_inventory_value_estimated if average_inventory_value_estimated else None
            ),
            "days_inventory_outstanding_estimated": days_inventory_outstanding(
                average_inventory_value_estimated, cogs, 365
            ),
            "holding_cost_estimated_annual": inventory_holding_cost(
                average_inventory_value_estimated, float(os.getenv("HOLDING_COST_RATE", "0.25"))
            ),
            "gross_profit_estimated": revenue - cogs,
            "stockout_rate_proxy_pct": (
                zero_stock_count / inventory_rows * 100 if inventory_rows else None
            ),
            "overstock_value_estimated": overstock_value,
            "units_sold": units_sold,
            "currency": "GBP",
            "inventory_source": "synthetic demonstration snapshot; final value is a proxy for period-average inventory",
            "cost_source": "synthetic estimated unit costs",
        }

    @app.get("/api/sales/monthly")
    def monthly_sales(request: Request, start_date: date | None = None, end_date: date | None = None):
        if request.app.state.engine.dialect.name == "mysql":
            month = func.date_format(DailySale.sales_date, "%Y-%m")
        else:
            month = func.strftime("%Y-%m", DailySale.sales_date)
        statement = select(
            month.label("month"),
            func.sum(DailySale.revenue).label("revenue"),
            func.sum(DailySale.units_sold).label("units"),
            func.sum(DailySale.estimated_cogs).label("estimated_cogs"),
        )
        if start_date:
            statement = statement.where(DailySale.sales_date >= start_date)
        if end_date:
            statement = statement.where(DailySale.sales_date <= end_date)
        rows = _read(request, statement.group_by(month).order_by(month))
        previous = None
        for row in rows:
            revenue = float(row["revenue"] or 0)
            cogs = float(row["estimated_cogs"] or 0)
            row["estimated_gross_profit"] = revenue - cogs
            row["revenue_growth_pct"] = (
                (revenue - previous) / previous * 100 if previous not in (None, 0) else None
            )
            previous = revenue
        return rows

    @app.get("/api/sales/products")
    def product_sales(
        request: Request,
        limit: int = Query(20, ge=1, le=100),
        start_date: date | None = None,
        end_date: date | None = None,
    ):
        revenue = func.sum(OrderItem.line_revenue).label("revenue")
        units = func.sum(OrderItem.quantity).label("units")
        statement = select(Product.product_id, Product.description, revenue, units).join(
            OrderItem, OrderItem.product_id == Product.product_id
        ).join(Order, Order.order_id == OrderItem.order_id)
        if start_date:
            statement = statement.where(Order.order_date >= start_date)
        if end_date:
            statement = statement.where(Order.order_date < pd.Timestamp(end_date) + pd.Timedelta(days=1))
        statement = statement.group_by(Product.product_id, Product.description).order_by(revenue.desc()).limit(limit)
        return _read(request, statement)

    @app.get("/api/sales/categories")
    def category_sales(request: Request, start_date: date | None = None, end_date: date | None = None):
        revenue = func.sum(OrderItem.line_revenue).label("revenue")
        units = func.sum(OrderItem.quantity).label("units")
        statement = select(
            Category.category_name,
            revenue,
            units,
        ).join(Product, Product.category_id == Category.category_id).join(
            OrderItem, OrderItem.product_id == Product.product_id
        ).join(Order, Order.order_id == OrderItem.order_id)
        if start_date:
            statement = statement.where(Order.order_date >= start_date)
        if end_date:
            statement = statement.where(Order.order_date < pd.Timestamp(end_date) + pd.Timedelta(days=1))
        statement = statement.group_by(Category.category_id, Category.category_name).order_by(revenue.desc())
        return _read(request, statement)

    @app.get("/api/sales/regions")
    def region_sales(
        request: Request,
        limit: int = Query(20, ge=1, le=100),
        start_date: date | None = None,
        end_date: date | None = None,
    ):
        revenue = func.sum(OrderItem.line_revenue).label("revenue")
        statement = select(Order.customer_country.label("country"), revenue).join(
            OrderItem, OrderItem.order_id == Order.order_id
        )
        if start_date:
            statement = statement.where(Order.order_date >= start_date)
        if end_date:
            statement = statement.where(Order.order_date < pd.Timestamp(end_date) + pd.Timedelta(days=1))
        statement = statement.group_by(Order.customer_country).order_by(revenue.desc()).limit(limit)
        return _read(request, statement)

    @app.get("/api/sales/history/{product_id}")
    def sales_history(request: Request, product_id: str, days: int = Query(180, ge=7, le=730)):
        units = func.sum(DailySale.units_sold).label("units_sold")
        statement = select(DailySale.sales_date, units).where(DailySale.product_id == product_id).group_by(
            DailySale.sales_date
        ).order_by(DailySale.sales_date.desc()).limit(days)
        frame = _frame(request, statement)
        if frame.empty:
            return []
        frame["sales_date"] = pd.to_datetime(frame.sales_date)
        return frame.sort_values("sales_date").assign(
            sales_date=lambda value: value.sales_date.dt.strftime("%Y-%m-%d")
        ).to_dict("records")

    @app.get("/api/customers/segments")
    def customer_segments(
        request: Request,
        start_date: date | None = None,
        end_date: date | None = None,
    ):
        statement = select(
            Order.customer_id.label("customer_id"),
            Order.order_id.label("invoice_id"),
            Order.order_date.label("invoice_date"),
            OrderItem.quantity.label("quantity"),
            OrderItem.unit_price.label("unit_price"),
        ).join(OrderItem, OrderItem.order_id == Order.order_id)
        if start_date:
            statement = statement.where(Order.order_date >= start_date)
        if end_date:
            statement = statement.where(Order.order_date < pd.Timestamp(end_date) + pd.Timedelta(days=1))
        frame = _frame(request, statement)
        if frame.empty:
            return {
                "summary": {
                    "total_identifiable_customers": 0,
                    "repeat_customer_rate": 0.0,
                    "average_customer_monetary_value": 0.0,
                    "purchase_frequency": 0.0,
                    "customer_revenue_contribution_pct": 0.0,
                },
                "segment_counts": {},
                "segment_revenue": {},
                "top_customers": [],
                "rfm_summary": [],
                "provenance": "Observed transactions are retained; missing customer IDs remain null and are excluded from customer-level segmentation.",
            }
        report = build_customer_segment_report(frame)
        rfm = report["rfm"].sort_values(["monetary", "frequency"], ascending=False)
        top_customers = rfm.head(10)[["customer_id", "recency_days", "frequency", "monetary", "segment"]].to_dict("records")
        return {
            "summary": {
                "total_identifiable_customers": int(report["summary"].get("total_identifiable_customers", 0)),
                "repeat_customer_rate": float(report["summary"].get("repeat_customer_rate", 0.0)),
                "average_customer_monetary_value": float(report["summary"].get("average_customer_monetary_value", 0.0)),
                "purchase_frequency": float(report["summary"].get("purchase_frequency", 0.0)),
                "customer_revenue_contribution_pct": float(report["summary"].get("customer_revenue_contribution_pct", 0.0)),
            },
            "segment_counts": report.get("segment_counts", {}),
            "segment_revenue": report.get("segment_revenue", {}),
            "top_customers": top_customers,
            "rfm_summary": rfm.to_dict("records"),
            "provenance": "Customer segmentation uses only valid observed transaction rows with non-null customer IDs; synthetic estimates are not treated as customer identities.",
        }

    @app.get("/api/data-quality")
    def data_quality(request: Request):
        order_rows = _frame(request, select(Order.order_id, Order.customer_id, Order.order_date))
        order_id_series = pd.Series(order_rows["order_id"].tolist()) if not order_rows.empty else pd.Series(dtype=str)
        missing_customer_count = int(_scalar(request, select(func.count()).where(Order.customer_id.is_(None))) or 0)
        invalid_quantities = int(_scalar(request, select(func.count()).where(OrderItem.quantity <= 0)) or 0)
        invalid_prices = int(_scalar(request, select(func.count()).where(OrderItem.unit_price < 0)) or 0)
        duplicate_order_ids = int(order_id_series.duplicated().sum()) if not order_rows.empty else 0
        refund_like_rows = int(_scalar(request, select(func.count()).where(OrderItem.quantity < 0)) or 0)
        return {
            "observed_orders": int(len(order_rows)),
            "duplicate_order_ids": duplicate_order_ids,
            "missing_customer_ids": missing_customer_count,
            "non_positive_quantity_rows": invalid_quantities,
            "negative_unit_price_rows": invalid_prices,
            "return_or_cancelled_rows_removed": refund_like_rows,
            "data_quality_score_pct": max(
                0.0,
                round(
                    100.0
                    - (
                        (duplicate_order_ids + missing_customer_count + invalid_quantities + invalid_prices + refund_like_rows)
                        / max(1, len(order_rows) or 1)
                        * 100.0
                    ),
                    2,
                ),
            ),
            "provenance": [
                "Observed transaction data comes from the UCI Online Retail II source dataset.",
                "Synthetic supplier, warehouse, and cost fields are kept separate from observed sales.",
                "Cancelled and return-like rows are excluded from customer-level analysis and demand metrics.",
            ],
        }

    @app.get("/api/inventory/summary")
    def inventory_summary(request: Request):
        rows = _stock_frame(request)
        if rows.empty:
            return {"available_stock": 0, "inventory_value": 0, "low_stock_products": 0, "overstock_products": 0, "dead_stock_value": 0}
        latest = _scalar(request, select(func.max(DailySale.sales_date)))
        if latest is None:
            return {"available_stock": int(rows.quantity_on_hand.sum()), "inventory_value": float(rows.inventory_value.sum()), "low_stock_products": 0, "overstock_products": 0, "dead_stock_value": 0, "inventory_source": "synthetic demonstration snapshot"}
        latest_date = pd.Timestamp(latest).date()
        demand = _demand_by_product(request, latest_date - pd.Timedelta(days=89))
        demand_rate = demand.set_index("product_id").units_sold.div(90).to_dict() if not demand.empty else {}
        last_sales = _read(
            request,
            select(DailySale.product_id, func.max(DailySale.sales_date).label("last_sale")).group_by(DailySale.product_id),
        )
        last_sale_by_product = {row["product_id"]: row["last_sale"] for row in last_sales}
        rows["daily_demand"] = rows.product_id.map(demand_rate).fillna(0)
        rows["coverage_days"] = rows.quantity_on_hand.div(rows.daily_demand.replace(0, pd.NA))
        low_days = int(os.getenv("LOW_STOCK_COVERAGE_DAYS", "14"))
        dead_days = int(os.getenv("DEAD_STOCK_DAYS", "180"))
        low = (rows.daily_demand > 0) & (rows.quantity_on_hand <= rows.daily_demand * low_days)
        over = (rows.daily_demand > 0) & (rows.coverage_days > int(os.getenv("TARGET_COVERAGE_DAYS", "45")) * float(os.getenv("RECOMMENDATION_STOCK_MULTIPLIER", "1.5")))
        last = pd.to_datetime(rows.product_id.map(last_sale_by_product))
        dead = rows.daily_demand.eq(0) & last.notna() & ((pd.Timestamp(latest_date) - last).dt.days >= dead_days)
        cogs = float(_scalar(request, select(func.sum(DailySale.estimated_cogs))) or 0)
        inventory_value = float(rows.inventory_value.sum())
        inventory_snapshot_values = rows["inventory_value"].tolist()
        average_inventory_value_estimated = float(np.mean(inventory_snapshot_values)) if inventory_snapshot_values else 0.0
        return {
            "available_stock": int(rows.quantity_on_hand.sum()),
            "inventory_value": inventory_value,
            "average_inventory_value_estimated": average_inventory_value_estimated,
            "low_stock_products": int(rows.loc[low, "product_id"].nunique()),
            "overstock_products": int(rows.loc[over, "product_id"].nunique()),
            "dead_stock_value": float(rows.loc[dead, "inventory_value"].sum()),
            "low_stock_coverage_days": low_days,
            "stockout_risk_coverage_days": int(os.getenv("STOCKOUT_RISK_COVERAGE_DAYS", "30")),
            "inventory_turnover_estimated": cogs / average_inventory_value_estimated if average_inventory_value_estimated else None,
            "days_inventory_outstanding_estimated": days_inventory_outstanding(average_inventory_value_estimated, cogs, 365),
            "holding_cost_estimated_annual": inventory_holding_cost(
                average_inventory_value_estimated, float(os.getenv("HOLDING_COST_RATE", "0.25"))
            ),
            "inventory_source": "synthetic demonstration snapshot; mean inventory value is a proxy across available inventory rows",
        }

    @app.get("/api/inventory/products")
    def inventory_products(request: Request, limit: int = Query(5000, ge=1, le=5000)):
        products_frame = _stock_frame(request)
        if products_frame.empty:
            return []
        latest = _scalar(request, select(func.max(DailySale.sales_date)))
        if latest is None:
            return []
        latest_date = pd.Timestamp(latest).date()
        avg = _demand_by_product(request, latest_date - pd.Timedelta(days=89))
        annual = _demand_by_product(request, latest_date - pd.Timedelta(days=364))
        cogs_statement = select(
            DailySale.product_id,
            func.sum(DailySale.estimated_cogs).label("annual_consumption_value"),
        ).where(DailySale.sales_date >= latest_date - pd.Timedelta(days=364)).group_by(DailySale.product_id)
        annual_value = _read(request, cogs_statement)
        demand_rate = avg.set_index("product_id").units_sold.div(90).to_dict() if not avg.empty else {}
        consumption = {row["product_id"]: float(row["annual_consumption_value"]) for row in annual_value}
        products_frame["daily_demand_90d"] = products_frame.product_id.map(demand_rate).fillna(0)
        products_frame["annual_consumption_value"] = products_frame.product_id.map(consumption).fillna(0)
        products_frame["annual_units_sold"] = products_frame.product_id.map(
            annual.set_index("product_id").units_sold.to_dict() if not annual.empty else {}
        ).fillna(0)
        ranked = products_frame.annual_consumption_value.clip(lower=0).sort_values(ascending=False)
        total = ranked.sum()
        cumulative = ranked.cumsum() / total if total else ranked * 0
        abc = pd.Series(
            ["A" if value <= 0.8 else "B" if value <= 0.95 else "C" for value in cumulative],
            index=ranked.index,
        )
        products_frame["abc_class"] = products_frame.product_id.map(abc).fillna("C")
        daily_variability = _daily_demand(
            request, latest_date - pd.Timedelta(days=364)
        )
        if not daily_variability.empty:
            daily_variability["sales_date"] = pd.to_datetime(daily_variability["sales_date"])
            dates = pd.date_range(latest_date - pd.Timedelta(days=364), latest_date, freq="D")
            demand_matrix = daily_variability.pivot(
                index="product_id", columns="sales_date", values="units_sold"
            ).reindex(columns=dates, fill_value=0)
            products_frame["xyz_class"] = products_frame.product_id.map(
                xyz_classification(demand_matrix)
            ).fillna("Z")
        else:
            products_frame["xyz_class"] = "Z"
        products_frame["inventory_coverage_days"] = products_frame.quantity_on_hand.div(
            products_frame.daily_demand_90d.replace(0, pd.NA)
        )
        products_frame["stockout_risk"] = products_frame.quantity_on_hand < products_frame.daily_demand_90d * int(
            os.getenv("STOCKOUT_RISK_COVERAGE_DAYS", "30")
        )
        result = products_frame.sort_values("annual_consumption_value", ascending=False).head(limit)
        return result.astype(object).where(pd.notna(result), None).to_dict("records")

    @app.get("/api/inventory/low-stock")
    def low_stock(request: Request, limit: int = Query(100, ge=1, le=1000)):
        return _inventory_alerts(request, "low", limit)

    @app.get("/api/inventory/overstock")
    def overstock(request: Request, limit: int = Query(100, ge=1, le=1000)):
        return _inventory_alerts(request, "over", limit)

    @app.get("/api/suppliers/performance")
    def supplier_performance(request: Request):
        statement = select(
            Supplier.supplier_id,
            Supplier.supplier_name,
            SupplierDelivery.lead_time_days,
            SupplierDelivery.on_time,
        ).join(SupplierDelivery, SupplierDelivery.supplier_id == Supplier.supplier_id)
        frame = _frame(request, statement)
        if frame.empty:
            return []
        result = frame.groupby(["supplier_id", "supplier_name"], as_index=False).agg(
            average_lead_time_days=("lead_time_days", "mean"),
            lead_time_std_days=("lead_time_days", "std"),
            on_time_delivery_pct=("on_time", lambda values: values.mean() * 100),
            deliveries=("on_time", "size"),
        )
        result["lead_time_std_days"] = result["lead_time_std_days"].fillna(0)
        result["data_source"] = "synthetic demonstration deliveries"
        return result.sort_values("on_time_delivery_pct").round(2).to_dict("records")

    @app.get("/api/forecast/{product_id}")
    def product_forecast(request: Request, product_id: str, horizon_days: int = Query(90, ge=7, le=90)):
        rows = _frame(
            request,
            select(DailySale.product_id, DailySale.sales_date, DailySale.units_sold).where(
                DailySale.product_id == product_id
            ),
        )
        if rows.empty:
            return {"product_id": product_id, "warning": "No database demand history is available.", "forecasts": []}
        result = forecast_product(
            rows, product_id, Path("models"), int(os.getenv("FORECAST_MIN_HISTORY_DAYS", "90")),
            int(os.getenv("FORECAST_VALIDATION_DAYS", "30")), horizon_days,
        )
        return {
            "product_id": product_id,
            "model_name": result.model_name,
            "validation_metrics": result.metrics,
            "forecasts": result.forecasts,
            "warning": result.warning,
            "uncertainty": result.uncertainty,
        }

    @app.get("/api/recommendations")
    def recommendations(
        request: Request,
        search: str | None = None,
        priority: str | None = Query(None, pattern="^(Critical|High|Medium|Low)$"),
        limit: int = Query(100, ge=1, le=5000),
    ):
        inv = _frame(
            request,
            select(
                Inventory.product_id,
                Product.description.label("product_name"),
                Product.supplier_id,
                func.sum(Inventory.quantity_on_hand).label("quantity_on_hand"),
                func.min(Inventory.data_source).label("data_source"),
            ).join(Product, Product.product_id == Inventory.product_id).group_by(
                Inventory.product_id, Product.description, Product.supplier_id
            ),
        )
        suppliers = pd.DataFrame(supplier_performance(request))
        if inv.empty:
            return []
        latest = _scalar(request, select(func.max(DailySale.sales_date)))
        daily = pd.DataFrame()
        history_starts: dict[str, object] = {}
        if latest is not None:
            latest_date = pd.Timestamp(latest).date()
            daily = _daily_demand(request, latest_date - pd.Timedelta(days=89))
            starts = _read(
                request,
                select(DailySale.product_id, func.min(DailySale.sales_date).label("first_sale")).group_by(DailySale.product_id),
            )
            history_starts = {row["product_id"]: row["first_sale"] for row in starts}
            last_sales = _read(
                request,
                select(DailySale.product_id, func.max(DailySale.sales_date).label("last_sale")).group_by(DailySale.product_id),
            )
            inv = inv.merge(pd.DataFrame(last_sales), on="product_id", how="left")
        output = generate_recommendations(
            inv,
            daily,
            suppliers,
            target_coverage_days=int(os.getenv("TARGET_COVERAGE_DAYS", "45")),
            stock_multiplier=float(os.getenv("RECOMMENDATION_STOCK_MULTIPLIER", "1.5")),
            service_z=float(os.getenv("SAFETY_STOCK_Z", "1.65")),
            minimum_history_days=int(os.getenv("FORECAST_MIN_HISTORY_DAYS", "90")),
            supplier_on_time_threshold_pct=float(os.getenv("SUPPLIER_ON_TIME_THRESHOLD_PCT", "80")),
            history_start_dates=history_starts,
            low_stock_coverage_days=int(os.getenv("LOW_STOCK_COVERAGE_DAYS", "14")),
            stockout_risk_coverage_days=int(os.getenv("STOCKOUT_RISK_COVERAGE_DAYS", "30")),
            slow_moving_days=int(os.getenv("SLOW_MOVING_DAYS", "90")),
            dead_stock_days=int(os.getenv("DEAD_STOCK_DAYS", "180")),
        )
        if search:
            output = [item for item in output if search.lower() in item["product_name"].lower() or search.lower() in item["product_id"].lower()]
        if priority:
            output = [item for item in output if item["priority"] == priority]
        return output[:limit]

    @app.get("/api/products")
    def products(
        request: Request,
        search: str | None = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=5000),
    ):
        statement = select(
            Product.product_id,
            Product.description,
            Category.category_name,
            Product.unit_price,
            Product.data_source,
        ).join(Category, Category.category_id == Product.category_id)
        total_statement = select(func.count(Product.product_id)).join(
            Category, Category.category_id == Product.category_id
        )
        if search:
            predicate = or_(Product.description.ilike(f"%{search}%"), Product.product_id.ilike(f"%{search}%"))
            statement = statement.where(predicate)
            total_statement = total_statement.where(predicate)
        rows = _read(request, statement.order_by(Product.product_id).offset(offset).limit(limit))
        total = int(_scalar(request, total_statement) or 0)
        return {"items": rows, "offset": offset, "limit": limit, "total": total}

    @app.get("/api/warehouses")
    def warehouses(request: Request, start_date: date | None = None, end_date: date | None = None):
        warehouse_rows = _frame(request, select(Warehouse.warehouse_id, Warehouse.warehouse_name, Warehouse.region))
        stock_rows = _read(
            request,
            select(Inventory.warehouse_id, func.sum(Inventory.quantity_on_hand).label("stock")).group_by(Inventory.warehouse_id),
        )
        sales_statement = select(
            DailySale.warehouse_id, func.sum(DailySale.revenue).label("revenue")
        )
        if start_date:
            sales_statement = sales_statement.where(DailySale.sales_date >= start_date)
        if end_date:
            sales_statement = sales_statement.where(DailySale.sales_date <= end_date)
        sales_rows = _read(request, sales_statement.group_by(DailySale.warehouse_id))
        if warehouse_rows.empty:
            return []
        stock = pd.DataFrame(stock_rows)
        revenue = pd.DataFrame(sales_rows)
        result = warehouse_rows.merge(stock, on="warehouse_id", how="left").merge(revenue, on="warehouse_id", how="left").fillna(0)
        result["data_source"] = "synthetic warehouse assignment; observed sales revenue"
        return result.to_dict("records")

    return app


def _inventory_alerts(request: Request, alert_type: str, limit: int) -> list[dict]:
    rows = _stock_frame(request)
    latest = _scalar(request, select(func.max(DailySale.sales_date)))
    if rows.empty or latest is None:
        return []
    start_date = pd.Timestamp(latest).date() - pd.Timedelta(days=89)
    demand = _demand_by_product(request, start_date)
    rate = demand.set_index("product_id").units_sold.div(90).to_dict() if not demand.empty else {}
    rows["daily_demand"] = rows.product_id.map(rate).fillna(0)
    rows["inventory_coverage_days"] = rows.quantity_on_hand.div(rows.daily_demand.replace(0, pd.NA))
    if alert_type == "low":
        low_days = int(os.getenv("LOW_STOCK_COVERAGE_DAYS", "14"))
        rows = rows[(rows.daily_demand > 0) & (rows.quantity_on_hand <= rows.daily_demand * low_days)]
    else:
        target = int(os.getenv("TARGET_COVERAGE_DAYS", "45")) * float(os.getenv("RECOMMENDATION_STOCK_MULTIPLIER", "1.5"))
        rows = rows[(rows.daily_demand > 0) & (rows.inventory_coverage_days > target)]
    result = rows.sort_values("inventory_coverage_days", ascending=alert_type == "low").head(limit)
    return result.astype(object).where(pd.notna(result), None).to_dict("records")


app = create_app()
