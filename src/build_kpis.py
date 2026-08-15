import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "business_docs"
OUTPUT_FILE = OUTPUT_DIR / "kpis.json"

KPIS = [

    {
        "name": "Total Store Sales",
        "table": "store_sales",
        "measure": "ss_net_paid",
        "aggregation": "SUM",
        "description": "Total revenue generated from physical store sales.",
        "keywords": [
            "sales",
            "revenue",
            "turnover",
            "store"
        ]
    },

    {
        "name": "Average Store Sales",
        "table": "store_sales",
        "measure": "ss_net_paid",
        "aggregation": "AVG",
        "description": "Average revenue per store sale.",
        "keywords": [
            "average",
            "sales",
            "store"
        ]
    },

    {
        "name": "Total Quantity Sold",
        "table": "store_sales",
        "measure": "ss_quantity",
        "aggregation": "SUM",
        "description": "Total quantity of products sold.",
        "keywords": [
            "quantity",
            "items sold",
            "products"
        ]
    },

    {
        "name": "Total Catalog Sales",
        "table": "catalog_sales",
        "measure": "cs_net_paid",
        "aggregation": "SUM",
        "description": "Total revenue from catalog sales.",
        "keywords": [
            "catalog",
            "sales",
            "revenue"
        ]
    },

    {
        "name": "Total Web Sales",
        "table": "web_sales",
        "measure": "ws_net_paid",
        "aggregation": "SUM",
        "description": "Total revenue from web sales.",
        "keywords": [
            "web",
            "online",
            "sales"
        ]
    },

    {
        "name": "Total Store Returns",
        "table": "store_returns",
        "measure": "sr_return_amt",
        "aggregation": "SUM",
        "description": "Total amount returned in physical stores.",
        "keywords": [
            "returns",
            "refund",
            "store"
        ]
    },

    {
        "name": "Total Catalog Returns",
        "table": "catalog_returns",
        "measure": "cr_return_amount",
        "aggregation": "SUM",
        "description": "Total amount returned from catalog purchases.",
        "keywords": [
            "catalog",
            "returns"
        ]
    },

    {
        "name": "Total Web Returns",
        "table": "web_returns",
        "measure": "wr_return_amt",
        "aggregation": "SUM",
        "description": "Total amount returned from web purchases.",
        "keywords": [
            "web",
            "returns"
        ]
    },

    {
        "name": "Inventory Quantity",
        "table": "inventory",
        "measure": "inv_quantity_on_hand",
        "aggregation": "SUM",
        "description": "Total quantity currently available in inventory.",
        "keywords": [
            "inventory",
            "stock"
        ]
    },

    {
        "name": "Customer Count",
        "table": "customer",
        "measure": "c_customer_sk",
        "aggregation": "COUNT",
        "description": "Total number of customers.",
        "keywords": [
            "customers",
            "count"
        ]
    },

    {
        "name": "Store Count",
        "table": "store",
        "measure": "s_store_sk",
        "aggregation": "COUNT",
        "description": "Number of stores.",
        "keywords": [
            "stores",
            "count"
        ]
    },

    {
        "name": "Product Count",
        "table": "item",
        "measure": "i_item_sk",
        "aggregation": "COUNT",
        "description": "Number of products.",
        "keywords": [
            "products",
            "items"
        ]
    },

    {
        "name": "Promotion Count",
        "table": "promotion",
        "measure": "p_promo_sk",
        "aggregation": "COUNT",
        "description": "Number of promotions.",
        "keywords": [
            "promotion",
            "campaign"
        ]
    },

    {
        "name": "Warehouse Count",
        "table": "warehouse",
        "measure": "w_warehouse_sk",
        "aggregation": "COUNT",
        "description": "Number of warehouses.",
        "keywords": [
            "warehouse"
        ]
    },

    {
        "name": "Average Discount",
        "table": "store_sales",
        "measure": "ss_ext_discount_amt",
        "aggregation": "AVG",
        "description": "Average discount amount applied to store sales.",
        "keywords": [
            "discount",
            "average"
        ]
    }

]


def build_kpis():

    result = []

    for idx, kpi in enumerate(KPIS, start=1):

        result.append({

            "id": f"KPI{idx:03d}",

            "name": kpi["name"],

            "description": kpi["description"],

            "table": kpi["table"],

            "measure": kpi["measure"],

            "aggregation": kpi["aggregation"],

            "formula": f"{kpi['aggregation']}({kpi['measure']})",

            "keywords": kpi["keywords"]

        })

    return result


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    kpis = build_kpis()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(
            kpis,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("=" * 60)
    print(f"KPIs generated : {len(kpis)}")
    print(f"Saved to : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()