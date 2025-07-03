# src/standardize_domain.py

import pandas as pd

def standardize_code(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    return df_copy



# 테스트 코드
if __name__ == "__main__":
    print("✅ 성별 정규화 테스트")
    df_customer = pd.read_csv("data/customer_info.csv")
    print(df_customer[["customer_id", "gender"]])
    df_std = standardize_gender(df_customer)
    print(df_std[["customer_id", "gender"]])