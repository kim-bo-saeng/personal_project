# src/infer_meta.py

import pandas as pd

def infer_customer_meta(df: pd.DataFrame) -> list:
    expected_columns = ["customer_id", "name", "age", "gender", "address", "signup_date"]
    missing_cols = [col for col in expected_columns if col not in df.columns]
    return missing_cols

def infer_order_meta(df: pd.DataFrame) -> list:
    expected_columns = ["order_id", "customer_id", "product_name", "quantity", "price", "status", "category"]
    missing_cols = [col for col in expected_columns if col not in df.columns]
    return missing_cols

# 테스트 코드
if __name__ == "__main__":
    df_customer = pd.read_csv("data/customer_info.csv")
    df_order = pd.read_csv("data/order_info.csv")

    print("🧩 고객 데이터 누락 메타:", infer_customer_meta(df_customer))
    print("🧩 주문 데이터 누락 메타:", infer_order_meta(df_order))