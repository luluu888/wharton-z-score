
import argparse
import sys
 
import pandas as pd
 
# ---------------------------------------------------------------------------
# THE ONE BLOCK YOU ARE ALLOWED TO ARGUE ABOUT -- BEFORE YOU SEE ANY OUTPUT.
# Weights are set in advance and do not move to make a preferred name rank
# higher. If you want to test sensitivity, copy this file and label the copy
# a sensitivity test.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "roic": 0.30,            # efficiency
    "fcf_yield": 0.25,       # cheapness
    "rev_cagr_3y": 0.25,     # growth
    "net_debt_ebitda": 0.20, # survival
}
 
# Metrics where a LOWER raw value is better. Their z-score gets negated so
# that "positive is good" holds for all four.
LOWER_IS_BETTER = {"net_debt_ebitda"}
 
WINSOR_LIMITS = (0.05, 0.95)  # clip to the 5th/95th percentile within sector
MIN_PER_SECTOR = 5            # below this, z-scores are too noisy to trust
 
 
def winsorize(s: pd.Series) -> pd.Series:
    """Clip a sector's values to its own 5th/95th percentile.
 
    The percentiles are computed on the RAW values -- that is what decides the
    clip thresholds. One company with a 400% ROIC accounting quirk would
    otherwise drag its sector's mean up and inflate its standard deviation,
    which distorts the score of every other company in that sector.
    """
    lo, hi = s.quantile(WINSOR_LIMITS[0]), s.quantile(WINSOR_LIMITS[1])
    return s.clip(lower=lo, upper=hi)
 
 
def zscore(s: pd.Series) -> pd.Series:
    """Standardize already-winsorized values.
 
    Note the order: the mean and standard deviation here are computed on the
    CLIPPED series, not on the raw one. Computing stats on raw data and then
    clipping is the classic version of this bug -- the outlier still poisons
    the mean, it just stops being visible.
 
    ddof=1 (pandas default) is the sample standard deviation, matching Excel's
    STDEV / STDEV.S.
    """
    sd = s.std()
    if sd == 0 or pd.isna(sd):   # guard: a sector where everyone is identical
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd
 
 
def score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    by_sector = df.groupby("sector", group_keys=False)
 
    for metric in WEIGHTS:
        # Step 1: winsorize within sector.
        wins = by_sector[metric].transform(winsorize)
        df[f"{metric}_wins"] = wins
 
        # Step 2: z-score the winsorized values, again within sector.
        z = df.groupby("sector", group_keys=False)[f"{metric}_wins"].transform(zscore)
 
        # Step 3: flip the sign where lower is better.
        if metric in LOWER_IS_BETTER:
            z = -z
        df[f"z_{metric}"] = z
 
    # Step 4: weighted sum.
    df["score"] = sum(df[f"z_{m}"] * w for m, w in WEIGHTS.items())
 
    # Step 5: rank, 1 = best.
    df["rank"] = df["score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("rank")
 
 
def validate(df: pd.DataFrame, top_n: int) -> list[str]:
    """The two checks that catch most bugs. Returns a list of warnings."""
    warnings = []
 
    # Check 0: thin sectors. A sector of 2 produces meaningless z-scores.
    counts = df["sector"].value_counts()
    thin = counts[counts < MIN_PER_SECTOR]
    for sector, n in thin.items():
        warnings.append(f"sector '{sector}' has only {n} companies -- z-scores unreliable")
 
    # Check 1: within each sector, each metric's z-scores must average ~0.
    # A sector that averages, say, +0.4 almost always means a label mismatch:
    # 'Tech' in some rows and 'Technology' in others, so they were scored as
    # two separate one-company sectors.
    for metric in WEIGHTS:
        means = df.groupby("sector")[f"z_{metric}"].mean()
        off = means[means.abs() > 0.05]
        for sector, m in off.items():
            warnings.append(f"mean z_{metric} for '{sector}' is {m:+.3f}, expected ~0 "
                            f"-- check for inconsistent sector labels")
 
    # Check 2: the top of the list shouldn't be one sector wearing a trenchcoat.
    # If it is, the sector-relative logic isn't actually neutralizing sector
    # effects and the ranking has become an undeclared sector bet.
    head = df.nsmallest(10, "rank")
    conc = head["sector"].value_counts()
    if len(head) and conc.iloc[0] / len(head) >= 0.6:
        warnings.append(f"{conc.iloc[0]} of the top {len(head)} are '{conc.index[0]}' "
                        f"-- sector-relative logic may not be working")
 
    # Sanity check on the inputs themselves: Net Debt/EBITDA is meaningless
    # when EBITDA is negative, and a negative ratio then reads as a fortress
    # balance sheet when the truth is the opposite.
    neg = df[df["net_debt_ebitda"] < 0]
    if len(neg):
        warnings.append(f"{len(neg)} companies have negative net_debt_ebitda -- confirm "
                        f"these are net-cash, not negative EBITDA: "
                        f"{', '.join(neg['ticker'].astype(str).head(5))}")
    return warnings
 
 
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--top", type=int, default=20, help="size of the shortlist")
    ap.add_argument("--out", default=None, help="write full ranked table to this CSV")
    args = ap.parse_args()
 
    df = pd.read_csv(args.csv)
 
    required = ["company", "ticker", "sector", *WEIGHTS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"missing required column(s): {', '.join(missing)}")
 
    # Rows with a missing metric cannot be scored. Dropping them silently is
    # how a company vanishes from a shortlist without anyone noticing.
    incomplete = df[df[list(WEIGHTS)].isna().any(axis=1)]
    if len(incomplete):
        print(f"WARNING: dropping {len(incomplete)} rows with missing data: "
              f"{', '.join(incomplete['ticker'].astype(str))}\n")
        df = df.dropna(subset=list(WEIGHTS))
 
    # Sector labels are the single most common source of silent error.
    df["sector"] = df["sector"].str.strip()
 
    ranked = score(df)
 
    for w in validate(ranked, args.top):
        print(f"WARNING: {w}")
    print()
 
    cols = ["rank", "ticker", "company", "sector", "score",
            "z_roic", "z_fcf_yield", "z_rev_cagr_3y", "z_net_debt_ebitda"]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(f"--- top {args.top} of {len(ranked)} ---")
        print(ranked.head(args.top)[cols].to_string(index=False,
              formatters={c: "{:+.2f}".format for c in cols if c.startswith(("z_", "score"))}))
 
    if args.out:
        ranked.to_csv(args.out, index=False)
        print(f"\nfull ranked table written to {args.out}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())