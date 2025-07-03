# src/standardize_domain.py

import pandas as pd

def standardize_gender(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    gender_map = {
        "M": "남", "Male": "남", "남성": "남",
        "F": "여", "Female": "여", "여성": "여",
        "N": None, "": None, None: None
    }

    df_copy["gender"] = df_copy["gender"].map(lambda g: gender_map.get(g, None))
    return df_copy


def standardize_status(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    status_map = {
        "배송완료": "배송완료",
        "완료": "배송완료",
        "배송중": "배송중",
        "Cancelled": "취소",
        "배송취소": "취소",
        "": None,
    }

    df_copy["status"] = df_copy["status"].map(lambda s: status_map.get(s, None))
    return df_copy


# 테스트 코드
if __name__ == "__main__":
    print("✅ 성별 정규화 테스트")
    df_customer = pd.read_csv("data/customer_info.csv")
    print(df_customer[["customer_id", "gender"]])
    df_std = standardize_gender(df_customer)
    print(df_std[["customer_id", "gender"]])

    print("\n✅ 주문 상태 정규화 테스트")
    df_order = pd.read_csv("data/order_info.csv")
    print(df_order[["order_id", "status"]])
    df_order_std = standardize_status(df_order)
    print(df_order_std[["order_id", "status"]])