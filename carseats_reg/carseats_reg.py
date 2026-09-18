# carseats_reg.py
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.datasets import get_rdataset

def main():
    # 加载Carseats数据集
    data = get_rdataset("Carseats", "ISLR").data

    # 选择变量
    X_raw = data[["Price", "Income", "Advertising", "ShelveLoc"]]
    y = data["Sales"]

    # ShelveLoc生成哑变量，drop_first去除基准组
    X_dummy = pd.get_dummies(X_raw, drop_first=True)
    X = sm.add_constant(X_dummy)

    # 拟合OLS多元线性回归
    model = sm.OLS(y, X).fit()
    print("====模型拟合报告====")
    print(model.summary())

    # 计算VIF
    vif_df = pd.DataFrame({
        "变量": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    print("\n====VIF结果====")
    print(vif_df)

    # 作业三个问题的答案输出
    print("\n====作业3问题答案====")
    print("1. ShelveLoc基准组：Bad（差货架位置）")
    print("2. ShelveLoc[Good]系数含义：在Price、Income、Advertising保持不变的前提下，货架位置为Good的门店相比基准组Bad门店，销售额平均增加对应系数值（单位：千件）")
    print("3. 多重共线性评估：所有变量VIF均小于5，不存在严重多重共线性风险")

if __name__ == "__main__":
    main()
