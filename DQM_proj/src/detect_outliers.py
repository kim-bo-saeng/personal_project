# src/detect_outliers.py

import pandas as pd

def detect_outliers_customer(df: pd.DataFrame) -> dict:
    report = {}

    # 1. 음수 age 탐지
    report["negative_age"] = df[df["age"].fillna(0) < 0].to_dict(orient="records")

    # 2. 중복 name + age 조합 탐지 (ID는 다르지만 동일 인물로 추정)
    dup = df[df.duplicated(subset=["name", "age"], keep=False)]
    report["possible_duplicates"] = dup.to_dict(orient="records")

    # 3. gender가 기준값(M/F/남/여 등)에 안 맞는 값 탐지
    allowed = {"M", "F", "남", "여", "Male", "여성"}
    report["unrecognized_gender"] = df[~df["gender"].isin(allowed)].to_dict(orient="records")

    return report


# 테스트 코드
if __name__ == "__main__":
    df = pd.read_csv("data/customer_info.csv")
    result = detect_outliers_customer(df)

    for key, items in result.items():
        print(f"\n=== {key.upper()} ===")
        for row in items:
            print(row)
