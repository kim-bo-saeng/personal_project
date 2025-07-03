# src/impute_missing.py

import pandas as pd
from sklearn.impute import KNNImputer

def impute_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    # 숫자형 컬럼 선택 (age만 있음)
    numeric_cols = ["age"]

    # KNNImputer로 결측값 보정
    imputer = KNNImputer(n_neighbors=2)
    df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])

    return df_copy


def impute_order_data(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    numeric_cols = ["quantity", "price"]

    # 수량이나 가격이 음수면 결측으로 간주
    for col in numeric_cols:
        df_copy.loc[df_copy[col].fillna(0) <= 0, col] = None

    imputer = KNNImputer(n_neighbors=2)
    df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])

    return df_copy


# 테스트 코드
if __name__ == "__main__":
    print("📦 Customer Data 보정 전/후")
    df_customer = pd.read_csv("data/customer_info.csv")
    print(df_customer[["customer_id", "age"]])
    df_imputed = impute_customer_data(df_customer)
    print(df_imputed[["customer_id", "age"]])

    print("\n📦 Order Data 보정 전/후")
    df_order = pd.read_csv("data/order_info.csv")
    print(df_order[["order_id", "quantity", "price"]])
    df_order_imputed = impute_order_data(df_order)
    print(df_order_imputed[["order_id", "quantity", "price"]])