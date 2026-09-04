from pathlib import Path
import json
import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

print("Loading raw data...")

transactions = pd.read_csv(
    RAW_DIR / "Transactions_v2.csv"
)

fraud_flags = pd.read_csv(
    RAW_DIR / "FraudFlags_v2.csv"
)

with open(
    RAW_DIR / "ExchangeRates_v2.json",
    "r",
    encoding="utf-8"
) as f:
    exchange_json = json.load(f)

exchange_rates = pd.json_normalize(
    exchange_json
)

print(
    f"Transactions: {len(transactions)}"
)

print(
    f"Fraud labels: {len(fraud_flags)}"
)


# ============================================================
# 2. STRUCTURAL VALIDATION
# ============================================================

validation_log = []


def log_issue(source, issue, field, count):
    validation_log.append({
        "source": source,
        "issue": issue,
        "field": field,
        "count": int(count)
    })


def check_required_columns(
    source,
    df,
    required_columns
):
    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    for col in missing:
        log_issue(
            source,
            "Missing required column",
            col,
            1
        )

    if missing:
        raise ValueError(
            f"{source} missing required columns: {missing}"
        )


def check_nulls(
    source,
    df,
    columns
):
    for col in columns:

        count = int(
            df[col].isna().sum()
        )

        if count > 0:
            log_issue(
                source,
                "Null value",
                col,
                count
            )


def check_duplicate_key(
    source,
    df,
    key
):
    count = int(
        df[key].duplicated().sum()
    )

    if count > 0:
        log_issue(
            source,
            "Duplicate key",
            key,
            count
        )


check_required_columns(
    "Transactions",
    transactions,
    [
        "TransactionID",
        "CustomerID",
        "TransactionDate",
        "Amount",
        "Merchant",
        "Location",
        "Currency"
    ]
)

check_required_columns(
    "FraudFlags",
    fraud_flags,
    [
        "TransactionID",
        "IsFraud"
    ]
)

check_required_columns(
    "ExchangeRates",
    exchange_rates,
    [
        "Date",
        "Currency",
        "RateToUSD"
    ]
)

check_nulls(
    "Transactions",
    transactions,
    [
        "TransactionID",
        "CustomerID",
        "TransactionDate",
        "Amount",
        "Currency"
    ]
)

check_nulls(
    "FraudFlags",
    fraud_flags,
    [
        "TransactionID",
        "IsFraud"
    ]
)

check_nulls(
    "ExchangeRates",
    exchange_rates,
    [
        "Date",
        "Currency",
        "RateToUSD"
    ]
)

check_duplicate_key(
    "Transactions",
    transactions,
    "TransactionID"
)

check_duplicate_key(
    "FraudFlags",
    fraud_flags,
    "TransactionID"
)

validation_df = pd.DataFrame(
    validation_log,
    columns=[
        "source",
        "issue",
        "field",
        "count"
    ]
)


# ============================================================
# 3. DATE STANDARDIZATION
# ============================================================

transactions["TransactionDate"] = pd.to_datetime(
    transactions["TransactionDate"],
    errors="coerce"
)

exchange_rates["Date"] = pd.to_datetime(
    exchange_rates["Date"],
    errors="coerce"
)

invalid_transaction_dates = int(
    transactions["TransactionDate"]
    .isna()
    .sum()
)

invalid_fx_dates = int(
    exchange_rates["Date"]
    .isna()
    .sum()
)

if invalid_transaction_dates > 0:
    log_issue(
        "Transactions",
        "Invalid date",
        "TransactionDate",
        invalid_transaction_dates
    )

if invalid_fx_dates > 0:
    log_issue(
        "ExchangeRates",
        "Invalid date",
        "Date",
        invalid_fx_dates
    )


# ============================================================
# 4. SANITIZATION
# ============================================================

transactions["Merchant"] = (
    transactions["Merchant"]
    .astype(str)
    .str.strip()
)

transactions["Location"] = (
    transactions["Location"]
    .astype(str)
    .str.strip()
    .str.upper()
)

transactions["Currency"] = (
    transactions["Currency"]
    .astype(str)
    .str.strip()
    .str.upper()
)

exchange_rates["Currency"] = (
    exchange_rates["Currency"]
    .astype(str)
    .str.strip()
    .str.upper()
)

transactions["unknown_merchant_flag"] = (
    transactions["Merchant"]
    .str.lower()
    .eq("unknown")
    .astype(int)
)

transactions["unknown_location_flag"] = (
    transactions["Location"]
    .eq("XX")
    .astype(int)
)


# ============================================================
# 5. FX JOIN — DATE + CURRENCY
# ============================================================

exchange_rates = (
    exchange_rates
    .rename(
        columns={
            "Date": "TransactionDate"
        }
    )
)

tx = transactions.merge(
    exchange_rates[
        [
            "TransactionDate",
            "Currency",
            "RateToUSD"
        ]
    ],
    on=[
        "TransactionDate",
        "Currency"
    ],
    how="left"
)

tx["fx_match_status"] = np.where(
    tx["RateToUSD"].notna(),
    "Matched",
    "Missing Rate"
)

tx["fx_missing_flag"] = (
    tx["RateToUSD"]
    .isna()
    .astype(int)
)


# ============================================================
# 6. CURRENCY NORMALIZATION
# AmountUSD = Amount × RateToUSD
# ============================================================

tx["AmountUSD"] = (
    tx["Amount"]
    * tx["RateToUSD"]
)

fx_rate_exceptions = (
    tx[
        tx["fx_missing_flag"] == 1
    ]
    .copy()
)


# ============================================================
# 7. HIGH VALUE ANOMALY
# ============================================================

merchant_stats = (
    tx[
        tx["AmountUSD"].notna()
    ]
    .groupby("Merchant")["AmountUSD"]
    .agg(
        merchant_mean="mean",
        merchant_std="std",
        merchant_count="count"
    )
    .reset_index()
)

tx = tx.merge(
    merchant_stats,
    on="Merchant",
    how="left"
)

tx["merchant_std"] = (
    tx["merchant_std"]
    .fillna(0)
)

tx["high_value_flag"] = (
    tx["AmountUSD"].notna()
    &
    (
        tx["AmountUSD"]
        >
        (
            tx["merchant_mean"]
            +
            2 * tx["merchant_std"]
        )
    )
).astype(int)


# ============================================================
# 8. GEO-MISMATCH
# Case-aligned:
# confirmed currency mismatch OR XX location
# ============================================================

expected_currency_by_location = {
    "MY": "MYR",
    "SG": "SGD",
    "US": "USD",
    "UK": "GBP"
}

tx["expected_currency"] = (
    tx["Location"]
    .map(
        expected_currency_by_location
    )
)

tx["confirmed_currency_mismatch_flag"] = (
    tx["expected_currency"].notna()
    &
    (
        tx["Currency"]
        != tx["expected_currency"]
    )
).astype(int)

tx["geo_mismatch_flag"] = (
    (
        tx["confirmed_currency_mismatch_flag"] == 1
    )
    |
    (
        tx["unknown_location_flag"] == 1
    )
).astype(int)


# ============================================================
# 9. MERCHANT RISK
# ============================================================

merchant_frequency = (
    tx["Merchant"]
    .value_counts()
)

tx["merchant_frequency"] = (
    tx["Merchant"]
    .map(
        merchant_frequency
    )
)

tx["merchant_risk_flag"] = (
    tx["unknown_merchant_flag"]
    .astype(int)
)


# ============================================================
# 10. VELOCITY CHECK
# Same customer, same day
# ============================================================

tx["same_day_customer_count"] = (
    tx
    .groupby(
        [
            "CustomerID",
            "TransactionDate"
        ]
    )["TransactionID"]
    .transform("count")
)

tx["velocity_flag"] = (
    tx["same_day_customer_count"]
    >= 2
).astype(int)


# ============================================================
# 11. CONSOLIDATED REQUIRED RED FLAGS
# ============================================================

required_flag_columns = [
    "high_value_flag",
    "geo_mismatch_flag",
    "merchant_risk_flag",
    "velocity_flag"
]

tx["red_flag_count"] = (
    tx[required_flag_columns]
    .sum(axis=1)
)

tx["has_red_flag"] = (
    tx["red_flag_count"]
    > 0
).astype(int)


# ============================================================
# 12. EXPLAINABLE AUDIT SCORE
# ============================================================

tx["audit_risk_score"] = (
    3 * tx["high_value_flag"]
    + 3 * tx["geo_mismatch_flag"]
    + 2 * tx["merchant_risk_flag"]
    + 2 * tx["velocity_flag"]
    + 1 * tx["unknown_location_flag"]
)


# ============================================================
# 13. AUDIT REASON
# ============================================================

def build_risk_reason(row):

    reasons = []

    if row["high_value_flag"] == 1:
        reasons.append(
            "Unusually high value for merchant"
        )

    if (
        row[
            "confirmed_currency_mismatch_flag"
        ] == 1
    ):
        reasons.append(
            "Currency does not match transaction location"
        )

    if row["unknown_location_flag"] == 1:
        reasons.append(
            "Transaction location unavailable"
        )

    if row["merchant_risk_flag"] == 1:
        reasons.append(
            "Unknown merchant"
        )

    if row["velocity_flag"] == 1:
        reasons.append(
            "Multiple transactions for same customer on same day"
        )

    if not reasons:
        return "No red flag"

    return "; ".join(reasons)


tx["audit_reason"] = (
    tx.apply(
        build_risk_reason,
        axis=1
    )
)


# ============================================================
# 14. AUDIT ACTION
# ============================================================

def build_audit_action(row):

    actions = []

    if row["high_value_flag"] == 1:
        actions.append(
            "Inspect transaction support and approval"
        )

    if (
        row[
            "confirmed_currency_mismatch_flag"
        ] == 1
    ):
        actions.append(
            "Verify transaction location and currency rationale"
        )

    if row["unknown_location_flag"] == 1:
        actions.append(
            "Obtain location evidence"
        )

    if row["merchant_risk_flag"] == 1:
        actions.append(
            "Verify merchant identity and supporting documentation"
        )

    if row["velocity_flag"] == 1:
        actions.append(
            "Review transaction sequence and authorization"
        )

    if row["fx_missing_flag"] == 1:
        actions.append(
            "Resolve missing exchange-rate support"
        )

    if not actions:
        return "No immediate action"

    return "; ".join(actions)


tx["audit_action"] = (
    tx.apply(
        build_audit_action,
        axis=1
    )
)


# ============================================================
# 15. FRAUD LABELS — EVALUATION ONLY
# ============================================================

evaluation = tx.merge(
    fraud_flags,
    on="TransactionID",
    how="left",
    validate="one_to_one"
)

if evaluation["IsFraud"].isna().any():
    raise ValueError(
        "One or more transactions have no fraud label."
    )

y_true = (
    evaluation["IsFraud"]
    .astype(int)
)


# ============================================================
# 16. CUSTOMER FREQUENCY
# ============================================================

customer_frequency = (
    evaluation["CustomerID"]
    .value_counts()
)

evaluation["customer_frequency"] = (
    evaluation["CustomerID"]
    .map(
        customer_frequency
    )
)

evaluation["log_amount"] = np.log1p(
    evaluation["Amount"]
)


# ============================================================
# 17. ISOLATION FOREST
# ============================================================

isolation_features = [
    "log_amount",
    "merchant_frequency",
    "customer_frequency",
    "unknown_merchant_flag",
    "unknown_location_flag",
    "geo_mismatch_flag",
    "high_value_flag",
    "velocity_flag"
]

X_isolation = (
    evaluation[
        isolation_features
    ]
    .fillna(0)
)

isolation_results = []
isolation_predictions = {}

for contamination in [
    0.03,
    0.05,
    0.08,
    0.10,
    0.15
]:

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42
    )

    raw_pred = (
        model
        .fit_predict(
            X_isolation
        )
    )

    pred = (
        raw_pred == -1
    ).astype(int)

    isolation_predictions[
        contamination
    ] = pred

    isolation_results.append({
        "Contamination": contamination,
        "Flagged": int(
            pred.sum()
        ),
        "Precision": precision_score(
            y_true,
            pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            pred,
            zero_division=0
        )
    })

isolation_performance = (
    pd.DataFrame(
        isolation_results
    )
    .sort_values(
        [
            "F1",
            "Recall",
            "Precision"
        ],
        ascending=False
    )
)

best_isolation_row = (
    isolation_performance
    .iloc[0]
)

best_contamination = (
    best_isolation_row[
        "Contamination"
    ]
)

evaluation[
    "isolation_forest_flag"
] = (
    isolation_predictions[
        best_contamination
    ]
)


# ============================================================
# 18. K-MEANS ANOMALY DETECTION
# ============================================================

evaluation["location_frequency"] = (
    evaluation["Location"]
    .map(
        evaluation[
            "Location"
        ].value_counts()
    )
)

evaluation["currency_frequency"] = (
    evaluation["Currency"]
    .map(
        evaluation[
            "Currency"
        ].value_counts()
    )
)

kmeans_features = [
    "log_amount",
    "merchant_frequency",
    "customer_frequency",
    "same_day_customer_count",
    "unknown_merchant_flag",
    "unknown_location_flag",
    "geo_mismatch_flag",
    "high_value_flag",
    "merchant_risk_flag",
    "velocity_flag",
    "location_frequency",
    "currency_frequency"
]

X_kmeans = (
    evaluation[
        kmeans_features
    ]
    .fillna(0)
)

scaler = StandardScaler()

X_kmeans_scaled = (
    scaler.fit_transform(
        X_kmeans
    )
)

kmeans = KMeans(
    n_clusters=4,
    n_init=20,
    random_state=42
)

evaluation["kmeans_cluster"] = (
    kmeans.fit_predict(
        X_kmeans_scaled
    )
)

evaluation[
    "kmeans_anomaly_score"
] = (
    kmeans
    .transform(
        X_kmeans_scaled
    )
    .min(axis=1)
)


# ============================================================
# 19. K-MEANS THRESHOLD TESTING
# ============================================================

kmeans_results = []

for quantile in [
    0.80,
    0.85,
    0.90,
    0.92,
    0.95,
    0.97
]:

    cutoff = np.quantile(
        evaluation[
            "kmeans_anomaly_score"
        ],
        quantile
    )

    pred = (
        evaluation[
            "kmeans_anomaly_score"
        ]
        >= cutoff
    ).astype(int)

    kmeans_results.append({
        "Quantile": quantile,
        "Flagged": int(
            pred.sum()
        ),
        "Precision": precision_score(
            y_true,
            pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            pred,
            zero_division=0
        )
    })

kmeans_performance = (
    pd.DataFrame(
        kmeans_results
    )
    .sort_values(
        [
            "F1",
            "Recall",
            "Precision"
        ],
        ascending=False
    )
)


# ============================================================
# 20. FINAL K-MEANS PRIORITY FLAG
# Selected from prior evaluation: 0.85 quantile
# ============================================================

final_kmeans_quantile = 0.85

final_kmeans_cutoff = np.quantile(
    evaluation[
        "kmeans_anomaly_score"
    ],
    final_kmeans_quantile
)

evaluation[
    "kmeans_anomaly_flag"
] = (
    evaluation[
        "kmeans_anomaly_score"
    ]
    >= final_kmeans_cutoff
).astype(int)


# ============================================================
# 21. FINAL AUDIT PRIORITY
# ============================================================

def assign_audit_priority(row):

    if (
        row["has_red_flag"] == 1
        and
        row["kmeans_anomaly_flag"] == 1
    ):
        return "Critical"

    if row[
        "kmeans_anomaly_flag"
    ] == 1:
        return "High"

    if row[
        "has_red_flag"
    ] == 1:
        return "Medium"

    return "Low"


evaluation["audit_priority"] = (
    evaluation.apply(
        assign_audit_priority,
        axis=1
    )
)

priority_map = {
    "Critical": 1,
    "High": 2,
    "Medium": 3,
    "Low": 4
}

evaluation["priority_sort"] = (
    evaluation[
        "audit_priority"
    ]
    .map(
        priority_map
    )
)


# ============================================================
# 22. MODEL PERFORMANCE — ACTUAL OUTPUTS
# ============================================================

required_rules_pred = (
    evaluation[
        "has_red_flag"
    ]
    .astype(int)
)

isolation_pred = (
    evaluation[
        "isolation_forest_flag"
    ]
    .astype(int)
)

kmeans_pred = (
    evaluation[
        "kmeans_anomaly_flag"
    ]
    .astype(int)
)


def build_metric_row(
    approach,
    pred
):

    return {
        "Approach": approach,

        "Precision": precision_score(
            y_true,
            pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            pred,
            zero_division=0
        ),

        "Flagged": int(
            pred.sum()
        ),

        "Review_Rate": float(
            pred.mean()
        )
    }


model_performance = pd.DataFrame([
    build_metric_row(
        "Required Rules",
        required_rules_pred
    ),

    build_metric_row(
        "Isolation Forest",
        isolation_pred
    ),

    build_metric_row(
        "K-Means Anomaly",
        kmeans_pred
    )
])

model_performance[
    "Review_Rate_%"
] = (
    model_performance[
        "Review_Rate"
    ]
    * 100
).round(1)


# ============================================================
# 23. FINAL POWER BI OUTPUT
# ============================================================

transaction_risk_output = (
    evaluation.copy()
)


# ============================================================
# 24. FINAL SCHEMA CHECK
# ============================================================

required_output_columns = [
    "TransactionID",
    "CustomerID",
    "TransactionDate",
    "Amount",
    "Merchant",
    "Location",
    "Currency",

    "RateToUSD",
    "AmountUSD",
    "fx_match_status",
    "fx_missing_flag",

    "unknown_merchant_flag",
    "unknown_location_flag",

    "high_value_flag",
    "confirmed_currency_mismatch_flag",
    "geo_mismatch_flag",
    "merchant_risk_flag",
    "velocity_flag",

    "same_day_customer_count",
    "merchant_frequency",
    "customer_frequency",

    "red_flag_count",
    "has_red_flag",

    "isolation_forest_flag",
    "kmeans_cluster",
    "kmeans_anomaly_score",
    "kmeans_anomaly_flag",

    "audit_risk_score",
    "audit_priority",
    "priority_sort",
    "audit_reason",
    "audit_action",

    "IsFraud"
]

missing_output_columns = [
    col
    for col in required_output_columns
    if col not in transaction_risk_output.columns
]

if missing_output_columns:
    raise ValueError(
        "Missing output columns: "
        + str(
            missing_output_columns
        )
    )


# ============================================================
# 25. EXPORT OUTPUTS
# ============================================================

transaction_risk_output.to_csv(
    OUTPUT_DIR
    / "transaction_risk_output.csv",
    index=False
)

model_performance.to_csv(
    OUTPUT_DIR
    / "model_performance.csv",
    index=False
)

validation_df.to_csv(
    OUTPUT_DIR
    / "validation_log.csv",
    index=False
)

fx_rate_exceptions.to_csv(
    OUTPUT_DIR
    / "fx_rate_exceptions.csv",
    index=False
)

print(
    "Audit pipeline completed successfully."
)

print(
    "Required-rule flags:",
    int(
        required_rules_pred.sum()
    )
)

print(
    "K-Means priority transactions:",
    int(
        kmeans_pred.sum()
    )
)

print("\nModel performance:")
print(
    model_performance[
        [
            "Approach",
            "Precision",
            "Recall",
            "F1",
            "Flagged",
            "Review_Rate_%"
        ]
    ].to_string(
        index=False
    )
)
