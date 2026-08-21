import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# 1. Load the dataset
# -----------------------------

df = pd.read_csv("Q36_online_customer_orders.csv")

print("Original Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

# -----------------------------
# 2. Check missing values
# -----------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# 3. Check repeated Order IDs
# -----------------------------

print("\nRepeated Order IDs:")
print(df[df["order_id"].duplicated(keep=False)])

# -----------------------------
# 4. Remove repeated Order IDs
# -----------------------------

df = df.drop_duplicates(subset="order_id", keep="first")

print("\nDataset after removing duplicates:")
print(df.shape)

# -----------------------------
# 5. Handle missing values
# -----------------------------

# Fill missing age group with mode
df["age_group"] = df["age_group"].fillna(
    df["age_group"].mode()[0]
)

# Fill missing delivery frequency with median
df["delivery_frequency_per_month"] = df[
    "delivery_frequency_per_month"
].fillna(
    df["delivery_frequency_per_month"].median()
)

print("\nMissing Values after cleaning:")
print(df.isnull().sum())

# -----------------------------
# 6. Calculate Average Value Per Item
# -----------------------------

df["avg_value_per_item"] = (
    df["order_value_inr"] / df["item_count"]
)

print("\nAverage Value Per Item:")
print(df[[
    "order_id",
    "order_value_inr",
    "item_count",
    "avg_value_per_item"
]].head())

# -----------------------------
# 7. Create Customer Order
#    Frequency Groups
# -----------------------------

def frequency_group(value):

    if value <= 4:
        return "Low"

    elif value <= 8:
        return "Medium"

    else:
        return "High"


df["customer_order_frequency_group"] = (
    df["delivery_frequency_per_month"]
    .apply(frequency_group)
)

print("\nCustomer Order Frequency Groups:")
print(
    df[
        [
            "order_id",
            "delivery_frequency_per_month",
            "customer_order_frequency_group"
        ]
    ].head()
)

# -----------------------------
# 8. Compare Order Value
#    Across Age Groups
# -----------------------------

age_analysis = df.groupby("age_group")[
    "order_value_inr"
].agg(["count", "mean", "sum"])

age_analysis = age_analysis.sort_values(
    "mean",
    ascending=False
)

print("\nOrder Value by Age Group:")
print(age_analysis)

# -----------------------------
# 9. Compare Order Value
#    Across Product Categories
# -----------------------------

category_analysis = df.groupby(
    "product_category"
)["order_value_inr"].agg(
    ["count", "mean", "sum"]
)

category_analysis = category_analysis.sort_values(
    "mean",
    ascending=False
)

print("\nOrder Value by Product Category:")
print(category_analysis)

# -----------------------------
# 10. Analyze Frequency Groups
# -----------------------------

frequency_analysis = df.groupby(
    "customer_order_frequency_group"
)["order_value_inr"].agg(
    ["count", "mean", "sum"]
)

frequency_analysis = frequency_analysis.sort_values(
    "mean",
    ascending=False
)

print("\nOrder Value by Frequency Group:")
print(frequency_analysis)

# -----------------------------
# 11. Average Item Value
#     by Product Category
# -----------------------------

item_analysis = df.groupby(
    "product_category"
)["avg_value_per_item"].mean()

item_analysis = item_analysis.sort_values(
    ascending=False
)

print("\nAverage Value Per Item by Category:")
print(item_analysis)

# -----------------------------
# 12. Create Output Folder
# -----------------------------

os.makedirs("output", exist_ok=True)

# Save cleaned dataset
df.to_csv(
    "output/cleaned_customer_orders.csv",
    index=False
)

# -----------------------------
# 13. Box Plot
# -----------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="age_group",
    y="order_value_inr"
)

plt.title("Order Value Distribution by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Order Value (INR)")

plt.tight_layout()

plt.savefig(
    "output/order_value_boxplot.png"
)

plt.show()

# -----------------------------
# 14. Bar Chart
# -----------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="product_category",
    y="order_value_inr",
    estimator="mean"
)

plt.title("Average Order Value by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Average Order Value (INR)")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "output/category_order_value_barplot.png"
)

plt.show()

# -----------------------------
# 15. Frequency Group Chart
# -----------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="customer_order_frequency_group",
    y="order_value_inr",
    estimator="mean"
)

plt.title("Average Order Value by Customer Frequency Group")
plt.xlabel("Order Frequency Group")
plt.ylabel("Average Order Value (INR)")

plt.tight_layout()

plt.savefig(
    "output/frequency_group_barplot.png"
)

plt.show()

print("\nAnalysis completed successfully!")

print("\nFiles saved inside output folder.")