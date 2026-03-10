"""
CLV Assignment — Data Cleaning
================================
Cleans transactions_2016_2017.csv and saves the result as transactions_cleaned.csv.

Steps applied:
  1. encode_returns   — binary encode returned_to_shop_id → is_returned (0/1), drop original
  2. drop_duplicates  — single pass after encoding; catches raw duplicates and any rows
                        made identical by the return encoding
  3. encode_sizes     — map letter shoe sizes (XS/S/M/L/XL) to EU numeric equivalents
  4. derive_gross_revenue — add pre-discount full price column
  5. encode_heel_ordinal  — map prod_heel text categories to ordered integers

Usage
-----
Local (files in data/):
    python data_cleaning.py



Requirements: pandas, numpy  
"""

import pandas as pd
import numpy as np
import os



DATA_DIR = "data/"   # relative to the repo root

TRANSACTIONS_FILE = DATA_DIR + "transactions_2016_2017.csv"
TRAIN_FILE        = DATA_DIR + "customer_clv_train.csv"
TEST_FILE         = DATA_DIR + "customer_clv_test.csv"

OUTPUT_FILE       = DATA_DIR + "transactions_cleaned.csv"





# =============================================================================
# INTEGRITY CHECK
# =============================================================================

def check_integrity(transactions_path, train_path, test_path):
    """
    Verify files exist and the transactions file looks complete.
    Raises RuntimeError with a clear description if anything fails.
    Catches truncated uploads before any cleaning work is wasted.
    """
    issues = []

    for label, path in [("transactions", transactions_path),
                        ("train",        train_path),
                        ("test",         test_path)]:
        if not os.path.exists(path):
            issues.append(f"MISSING: {label} not found at {path}")

    if issues:
        raise RuntimeError("\n".join(issues))

    raw = pd.read_csv(transactions_path, low_memory=False)

    if len(raw) < 300_000:
        issues.append(f"LOW ROWS: {len(raw):,} rows (expected ≥300,000 — file may be truncated)")
    if raw['order_date'].max() < "2017-12-01":
        issues.append(f"TRUNCATED DATE: max order_date is {raw['order_date'].max()}")

    train_ids      = set(pd.read_csv(train_path)['cust_id'].astype(str))
    test_ids       = set(pd.read_csv(test_path)['cust_id'].astype(str))
    tx_ids         = set(raw['cust_id'].astype(str))
    train_miss_pct = len(train_ids - tx_ids) / len(train_ids)
    test_miss_pct  = len(test_ids  - tx_ids) / len(test_ids)

    if train_miss_pct > 0.01:
        issues.append(f"CUSTOMER MISMATCH: {train_miss_pct:.1%} of train customers not in transactions")
    if test_miss_pct > 0.01:
        issues.append(f"CUSTOMER MISMATCH: {test_miss_pct:.1%} of test customers not in transactions")

    if issues:
        raise RuntimeError(
            "Integrity check failed "
            "instead of uploading the file manually.\n\n"
            + "\n".join(f"  {i+1}. {v}" for i, v in enumerate(issues))
        )

    print(f"Integrity OK — {len(raw):,} rows, {raw['order_date'].min()} → {raw['order_date'].max()}")


# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================

def encode_returns(trx):
    """
    Binary encode returned_to_shop_id → is_returned (1 = returned, 0 = not returned).
    The original shop ID column is dropped, possible of investigation for report.
    Running this first collapses rows that only differ by returned_to_shop_id before
    dedup, so a single duplicate-removal pass handles everything.
    """
    trx = trx.copy()
    trx['is_returned'] = trx['returned_to_shop_id'].notna().astype(int)
    trx = trx.drop(columns=['returned_to_shop_id'])
    print(f"encode_returns    : return rate {trx['is_returned'].mean():.1%}")
    return trx


def drop_duplicates(trx):
    """
    Remove rows that are identical across all remaining columns.
    Single pass after encode_returns catches:
      - original exact duplicates (same across all 26 raw columns)
      - rows made identical by encoding (e.g. same item returned to two shops,
        which is a logging error — verified from the data: same sale_id, prod_id,
        revenue, date, and every product attribute; only shop ID differed)
    """
    before = len(trx)
    trx    = trx.drop_duplicates()
    print(f"drop_duplicates   : {before - len(trx):,} removed ({len(trx):,} remain)")
    return trx


def encode_sizes(trx):
    """
    Map letter-format prod_size values to EU numeric shoe sizes.
    XS maps to EU 17 (baby shoe) not EU 34 (adult small) — confirmed by cross-tabulating
    XS rows against prod_type_1, which shows XS appears exclusively in baby/boys products.
    """
    size_map = {'XS': 17, 'xs': 17,   # baby shoe — EU 17 (~0-6 months)
                'S':  36, 's':  36,   # adult EU small
                'M':  38, 'm':  38,   # adult EU medium
                'L':  40, 'l':  40,   # adult EU large
                'XL': 42, 'xl': 42}   # adult EU extra-large

    letter_mask = trx['prod_size'].astype(str).str.strip().isin(size_map.keys())
    if letter_mask.sum() > 0:
        cross = (trx.loc[letter_mask]
                    .assign(sz=trx.loc[letter_mask, 'prod_size'].astype(str).str.upper().str.strip())
                    .groupby(['sz', 'prod_type_1']).size().unstack(fill_value=0))
        print("encode_sizes      : letter size × product type cross-tab:")
        print(cross.to_string())

    trx['prod_size'] = trx['prod_size'].replace(size_map)
    trx['prod_size'] = pd.to_numeric(trx['prod_size'], errors='coerce')
    print(f"encode_sizes      : range {trx['prod_size'].min():.0f}–{trx['prod_size'].max():.0f}, "
          f"NaN remaining: {trx['prod_size'].isna().sum():,}")
    return trx


def derive_gross_revenue(trx):
    """
    Add gross_revenue = pre-discount full price.
    sale_discount_applied is stored as a negative number (the deduction applied),
    so subtracting it recovers the original price: sale_revenue - (-discount) = full price.
    Example: sale_revenue=83.30, sale_discount_applied=-35.70 → gross_revenue=119.00
    """
    trx = trx.copy()
    trx['gross_revenue'] = trx['sale_revenue'] - trx['sale_discount_applied']
    print(f"derive_gross_rev  : mean gross {trx['gross_revenue'].mean():.2f} "
          f"vs mean net {trx['sale_revenue'].mean():.2f}")
    return trx


def encode_heel_ordinal(trx):
    """
    Map prod_heel text categories to ordered integers.
    Ordinal: <2.5 cm=0, 2.5-5 cm=1, 5-8 cm=2, >8 cm=3
    NaN (flat shoes / no heel info) → -1, distinguishable from the lowest heel class.
    """
    heel_map = {'<2.5 cm': 0, '2.5-5 cm': 1, '5-8 cm': 2, '>8 cm': 3}
    trx = trx.copy()
    trx['prod_heel_ordinal'] = trx['prod_heel'].map(heel_map).fillna(-1).astype(int)
    print(f"encode_heel_ord   : {trx['prod_heel_ordinal'].value_counts().sort_index().to_dict()}")
    return trx


# =============================================================================
# PIPELINE
# =============================================================================

def clean(transactions_path, train_path, test_path, output_path):

    check_integrity(transactions_path, train_path, test_path)

    trx   = pd.read_csv(transactions_path, parse_dates=['order_date', 'pack_date'], low_memory=False)
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    print(f"\nLoaded: {len(trx):,} transactions, {trx['cust_id'].nunique():,} customers\n")

    trx = encode_returns(trx)         # binary encode returns, drop shop ID
    trx = drop_duplicates(trx)        # single dedup pass over all remaining columns
    trx = encode_sizes(trx)           # letter sizes → EU numeric
    trx = derive_gross_revenue(trx)   # add pre-discount full price
    trx = encode_heel_ordinal(trx)    # heel text → ordinal int

    print(f"\nFinal shape: {len(trx):,} rows × {len(trx.columns)} columns")
    print(f"Columns: {list(trx.columns)}")

    trx.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    return trx, train, test


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    trx, train, test = clean(TRANSACTIONS_FILE, TRAIN_FILE, TEST_FILE, OUTPUT_FILE)
