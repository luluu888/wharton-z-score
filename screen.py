
import argparse
import sys
 
import pandas as pd
 
# The weights live in weights.py, which is its own first commit in this
# repository -- made before any company data or results existed. They do not
# move to make a preferred name rank higher. To test sensitivity, copy the file
# and label the copy a sensitivity test.
from weights import LOWER_IS_BETTER, WEIGHTS  # noqa: E402
 
WINSOR_LIMITS = (0.05, 0.95)  # clip to the 5th/95th percentile within sector
MIN_PER_SECTOR = 5            # below this, z-scores are too noisy to trust
ZERO_VARIANCE_TOL = 1e-9      # relative: below this, treat the spread as zero
 
# How many companies a sector needs before its z-scores carry information.
# These are not taste -- they are arithmetic, and measurable:
#   n = 1  ->  z is exactly 0.000 (no peers to deviate from)
#   n = 2  ->  z is always +0.707 / -0.707 whatever the underlying values
#   n = 3  ->  winsorizing clips all three to the median, variance goes to
#              zero, and every z is exactly 0.000
#   n = 4-5 ->  real but coarse; one name moves the whole sector's mean
#   n >= 6  ->  usable
RELIABLE_PER_SECTOR = 6
ARTIFACT_PER_SECTOR = 3       # at or below this, scores are arithmetic, not measurement
 
 
def winsorize(s: pd.Series) -> pd.Series:
    """Clip a sector's values to its own 5th/95th percentile.
 
    The percentiles are computed on the RAW values -- that is what decides the
    clip thresholds. One company with a 400% ROIC accounting quirk would
    otherwise drag its sector's mean up and inflate its standard deviation,
    which distorts the score of every other company in that sector.
 
    On the interpolation arguments -- these matter more than they look.
 
    With a default (linear) percentile and only ~10 companies in a sector, the
    95th percentile lands BETWEEN the top two observations, so the outlier is
    roughly half of its own clip threshold. Push a company from 135% ROIC to
    4000% and the threshold rises with it: the clipping is cosmetic and the
    outlier still moves its peers' scores. Measured on the sample data, that
    setup protected peers no better than not winsorizing at all.
 
    Taking the inner observed value instead -- 'higher' at the bottom, 'lower'
    at the top -- puts the threshold on a real data point at least one rank
    inside the extreme. The outlier's magnitude then has no influence on
    anything: it is clipped to its neighbour's value no matter how absurd it is.
    """
    lo = s.quantile(WINSOR_LIMITS[0], interpolation="higher")
    hi = s.quantile(WINSOR_LIMITS[1], interpolation="lower")
    return s.clip(lower=lo, upper=hi)
 
 
def zscore(s: pd.Series) -> pd.Series:
    """Standardize already-winsorized values.
 
    Note the order: the mean and standard deviation here are computed on the
    CLIPPED series, not on the raw one. Computing stats on raw data and then
    clipping is the classic version of this bug -- the outlier still poisons
    the mean, it just stops being visible.
 
    ddof=1 (pandas default) is the sample standard deviation, matching Excel's
    STDEV / STDEV.S.
 
    On the tolerance: `sd == 0` is not enough. Values that are identical through
    arithmetic rather than being the same literal leave floating-point dust --
    a real case measured here came out at 3.0e-17, not 0. That passes an
    equality check, and dividing by it manufactured a z-score of +0.91 for
    every company in the sector out of pure rounding noise, which then flowed
    into the weighted score looking like signal. The comparison has to be a
    tolerance relative to the data's own scale.
    """
    sd = s.std()
    if pd.isna(sd) or sd <= ZERO_VARIANCE_TOL * max(1e-12, abs(s.mean())):
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
 
 
def sector_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sector count, best score, and whether that score can be trusted.
 
    Replaces counting sectors by hand with COUNTIF/MAXIFS. The verdict column
    answers one question only: is this sector's z-score arithmetic reliable?
 
    It deliberately does NOT decide whether to hold a sector. That call depends
    on the client -- their existing exposures, their income, their risk
    tolerance -- and none of that is in this data. Two further warnings about
    reading this table:
 
    Best scores are NOT comparable across sectors. A +1.5 in Technology and a
    +1.5 in Utilities each mean '1.5 standard deviations better than its own
    peers'. Neither says one sector is more attractive than the other -- that
    comparison is precisely what within-sector scoring removes. Picking which
    sectors to hold by comparing this column drops the sector whose peer group
    happened to be tightly clustered, not the weakest sector.
 
    A sector flagged ARTIFACT has a best score that is a number, not a
    measurement. Do not rank it, drop it, or keep it on that number.
    """
    rows = []
    for sector, grp in df.groupby("sector"):
        n = len(grp)
        # A sector is degenerate on a metric when its winsorized spread is
        # effectively zero, which forces every z-score in it to 0.
        degenerate = sum(
            1 for m in WEIGHTS
            if grp[f"{m}_wins"].std() <= ZERO_VARIANCE_TOL * max(1e-12, abs(grp[f"{m}_wins"].mean()))
            or pd.isna(grp[f"{m}_wins"].std())
        )
        if n <= ARTIFACT_PER_SECTOR:
            verdict, reason = "ARTIFACT", f"only {n} name(s); z-scores are arithmetic, not signal"
        elif n < RELIABLE_PER_SECTOR:
            verdict, reason = "WEAK", f"{n} names; one company moves the sector mean"
        elif degenerate:
            verdict, reason = "WEAK", f"{degenerate} of {len(WEIGHTS)} metrics have no spread"
        else:
            verdict, reason = "OK", ""
        rows.append({
            "sector": sector,
            "n": n,
            "best_score": grp["score"].max(),
            "median_score": grp["score"].median(),
            "best_ticker": grp.loc[grp["score"].idxmax(), "ticker"],
            "flat_metrics": degenerate,
            "verdict": verdict,
            "note": reason,
        })
    return pd.DataFrame(rows).sort_values(["verdict", "sector"])
 
 
def shortlist(df: pd.DataFrame, per_sector: int, exclude: set[str]) -> pd.DataFrame:
    """Top `per_sector` names from each sector, skipping excluded sectors.
 
    The per-sector cut is what keeps a shortlist from filling up with whichever
    sector happened to score well -- the same reason the scoring is
    within-sector in the first place.
    """
    kept = df[~df["sector"].isin(exclude)]
    picks = (kept.sort_values("score", ascending=False)
                 .groupby("sector", group_keys=False)
                 .head(per_sector))
    return picks.sort_values(["sector", "score"], ascending=[True, False])
 
 
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--top", type=int, default=20, help="size of the shortlist")
    ap.add_argument("--out", default=None, help="write full ranked table to this CSV")
    ap.add_argument("--sector-report", action="store_true",
                    help="per-sector counts, best score, and whether it can be trusted")
    ap.add_argument("--per-sector", type=int, default=None,
                    help="take this many names from each sector instead of a flat top-N")
    ap.add_argument("--exclude-sector", action="append", default=[],
                    help="sector to leave out (repeatable); record WHY in your report")
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
    fmt = {c: "{:+.2f}".format for c in cols if c.startswith(("z_", "score"))}
 
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        if args.sector_report:
            rep = sector_report(ranked)
            print("--- sector report ---")
            print(rep.to_string(index=False, formatters={
                "best_score": "{:+.2f}".format, "median_score": "{:+.2f}".format}))
            print("\nVerdict says only whether a sector's z-scores are reliable. Best scores "
                  "are NOT comparable between sectors -- that comparison is what "
                  "within-sector scoring removes. Which sectors to hold is a client "
                  "judgment, not an output of this table.\n")
 
        if args.per_sector:
            picks = shortlist(ranked, args.per_sector, set(args.exclude_sector))
            label = f"top {args.per_sector} per sector"
            if args.exclude_sector:
                label += f", excluding {', '.join(args.exclude_sector)}"
            print(f"--- {label}: {len(picks)} names ---")
            print(picks[cols].to_string(index=False, formatters=fmt))
        else:
            print(f"--- top {args.top} of {len(ranked)} ---")
            print(ranked.head(args.top)[cols].to_string(index=False, formatters=fmt))
 
    if args.out:
        ranked.to_csv(args.out, index=False)
        print(f"\nfull ranked table written to {args.out}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())