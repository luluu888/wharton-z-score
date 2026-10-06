import argparse
import sys
import time
from datetime import date
 
import pandas as pd
 
DEFAULT_TAX_RATE = 0.21   # US federal statutory, used when the effective rate is unusable
MAX_TAX_RATE = 0.50       # above this, the effective rate is an artifact, not a tax rate
CAGR_YEARS = 3
 
# Yahoo's statement row labels, in preference order. Yahoo renames these
# periodically, so each lookup tries several and records which one it used.
ALIASES = {
    "revenue":      ["Total Revenue", "Operating Revenue"],
    "ebit":         ["EBIT", "Operating Income", "Total Operating Income As Reported"],
    "ebitda":       ["EBITDA", "Normalized EBITDA"],
    "pretax":       ["Pretax Income", "Income Before Tax"],
    "tax":          ["Tax Provision", "Income Tax Expense"],
    "total_debt":   ["Total Debt"],
    "equity":       ["Stockholders Equity", "Total Equity Gross Minority Interest",
                     "Common Stock Equity"],
    "cash":         ["Cash And Cash Equivalents",
                     "Cash Cash Equivalents And Short Term Investments"],
    "ocf":          ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex":        ["Capital Expenditure", "Purchase Of PPE"],
    "dep_amort":    ["Depreciation And Amortization",
                     "Depreciation Amortization Depletion",
                     "Reconciled Depreciation"],
}
 
# Yahoo uses its own sector names. Mapped here to the GICS-style names the Excel
# template expects. Pass --raw-sectors to keep Yahoo's. What matters either way
# is that all 60 companies use ONE naming scheme -- a mix of 'Technology' and
# 'Tech' splits a sector in two and the screen's first check will flag it.
SECTOR_RENAME = {
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
    "Communication Services": "Communication Services",
    "Technology": "Technology",
    "Healthcare": "Healthcare",
    "Industrials": "Industrials",
    "Utilities": "Utilities",
    "Energy": "Energy",
    "Real Estate": "Real Estate",
}
 
 
# ---------------------------------------------------------------------------
# Pure metric math. No network, no yfinance -- all of this is unit-tested.
# ---------------------------------------------------------------------------
 
def effective_tax_rate(tax, pretax) -> float:
    """Effective rate, falling back to the statutory rate when unusable.
 
    A loss-making year gives a negative or absurd ratio (a tax benefit divided
    by a negative pretax income), which would turn NOPAT into nonsense.
    """
    if pd.isna(tax) or pd.isna(pretax) or pretax <= 0:
        return DEFAULT_TAX_RATE
    rate = tax / pretax
    if rate < 0 or rate > MAX_TAX_RATE:
        return DEFAULT_TAX_RATE
    return float(rate)
 
 
def roic(ebit, tax, pretax, total_debt, equity, cash):
    """NOPAT / invested capital.
 
    invested capital = total debt + total equity - cash. Subtracting cash is
    the choice that makes this comparable across companies holding very
    different cash piles; it is also why this will differ from sources that
    don't subtract it.
    """
    if pd.isna(ebit) or pd.isna(total_debt) or pd.isna(equity) or pd.isna(cash):
        return float("nan")
    invested = total_debt + equity - cash
    if invested <= 0:          # net cash exceeds capital employed -- ratio is meaningless
        return float("nan")
    nopat = ebit * (1 - effective_tax_rate(tax, pretax))
    return float(nopat / invested)
 
 
def fcf_yield(ocf, capex, market_cap, total_debt, cash):
    """(operating cash flow - capex) / enterprise value.
 
    Yahoo reports capex as a negative number, but not always, so abs() is used
    rather than addition. EV is market cap + total debt - cash.
    """
    if pd.isna(ocf) or pd.isna(capex) or pd.isna(market_cap):
        return float("nan")
    debt = 0.0 if pd.isna(total_debt) else total_debt
    csh = 0.0 if pd.isna(cash) else cash
    ev = market_cap + debt - csh
    if ev <= 0:
        return float("nan")
    fcf = ocf - abs(capex)
    return float(fcf / ev)
 
 
def rev_cagr(rev_latest, rev_prior, years: int = CAGR_YEARS):
    """Compound annual growth rate over `years` full fiscal years.
 
    Returns NaN when the base year is missing or non-positive -- a company
    that listed two years ago has no three-year growth rate, and inventing one
    is worse than dropping the row.
    """
    if pd.isna(rev_latest) or pd.isna(rev_prior) or rev_prior <= 0 or rev_latest <= 0:
        return float("nan")
    return float((rev_latest / rev_prior) ** (1 / years) - 1)
 
 
def net_debt_ebitda(total_debt, cash, ebitda):
    """(total debt - cash) / EBITDA.
 
    Deliberately returns NaN when EBITDA <= 0. The ratio flips sign there, so a
    loss-making company would score as having a fortress balance sheet. NaN
    makes screen.py drop the row and say so, which is the honest outcome.
    """
    if pd.isna(total_debt) or pd.isna(ebitda) or ebitda <= 0:
        return float("nan")
    csh = 0.0 if pd.isna(cash) else cash
    return float((total_debt - csh) / ebitda)
 
 
def compute_metrics(items: dict) -> dict:
    """Turn a dict of raw line items into the four screen inputs."""
    return {
        "roic": roic(items.get("ebit"), items.get("tax"), items.get("pretax"),
                     items.get("total_debt"), items.get("equity"), items.get("cash")),
        "fcf_yield": fcf_yield(items.get("ocf"), items.get("capex"),
                               items.get("market_cap"), items.get("total_debt"),
                               items.get("cash")),
        "rev_cagr_3y": rev_cagr(items.get("revenue"), items.get("revenue_prior")),
        "net_debt_ebitda": net_debt_ebitda(items.get("total_debt"), items.get("cash"),
                                          items.get("ebitda")),
    }
 
 
# ---------------------------------------------------------------------------
# Statement plumbing. Pure given a frame, so it is testable with fixtures.
# ---------------------------------------------------------------------------
 
def newest_first(frame) -> pd.DataFrame:
    """Sort statement columns (period end dates) newest first."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    return frame.reindex(sorted(frame.columns, reverse=True), axis=1)
 
 
def pick(frame: pd.DataFrame, key: str, col: int = 0):
    """Read one line item by trying each alias in turn.
 
    Returns (value, label_used). A missing item returns (NaN, None) -- never 0.
    """
    if frame.empty or col >= frame.shape[1]:
        return float("nan"), None
    for label in ALIASES[key]:
        if label in frame.index:
            value = frame.loc[label].iloc[col]
            if pd.notna(value):
                return float(value), label
    return float("nan"), None
 
 
def extract(inc: pd.DataFrame, bs: pd.DataFrame, cf: pd.DataFrame,
            market_cap=float("nan")) -> tuple[dict, list[str]]:
    """Pull every needed line item out of three statement frames.
 
    Returns (items, notes). Notes name anything that could not be found, so the
    caller can put it in the CSV instead of discovering it later.
    """
    inc, bs, cf = newest_first(inc), newest_first(bs), newest_first(cf)
    items, notes = {"market_cap": market_cap}, []
 
    for key, frame in [("revenue", inc), ("ebit", inc), ("pretax", inc), ("tax", inc),
                       ("total_debt", bs), ("equity", bs), ("cash", bs),
                       ("ocf", cf), ("capex", cf)]:
        value, label = pick(frame, key, 0)
        items[key] = value
        if label is None:
            notes.append(f"missing:{key}")
 
    # Revenue from `CAGR_YEARS` fiscal years earlier. Column 3 of an annual
    # statement is three years before column 0.
    items["revenue_prior"], label = pick(inc, "revenue", CAGR_YEARS)
    if label is None:
        notes.append(f"missing:revenue_prior(needs {CAGR_YEARS + 1} annual periods)")
 
    # EBITDA: reported row if present, otherwise rebuilt as EBIT + D&A.
    ebitda, label = pick(inc, "ebitda", 0)
    if label is None:
        da, da_label = pick(cf, "dep_amort", 0)
        if da_label is not None and pd.notna(items["ebit"]):
            ebitda = items["ebit"] + da
            notes.append("ebitda:derived from EBIT + D&A")
        else:
            notes.append("missing:ebitda")
    items["ebitda"] = ebitda
 
    if pd.isna(market_cap):
        notes.append("missing:market_cap")
    return items, notes
 
 
# ---------------------------------------------------------------------------
# The network layer. NOT unit-tested -- this is the part to suspect first.
# ---------------------------------------------------------------------------
 
def line_items(symbol: str, retries: int = 3, pause: float = 1.0):
    """Fetch one company's statements and identity from Yahoo.
 
    yfinance is imported here, not at module scope, so screen.py and the tests
    never need it installed.
    """
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed. Run: pip install yfinance")
 
    last_error = None
    for attempt in range(retries):
        try:
            t = yf.Ticker(symbol)
            info = t.get_info() or {}
            items, notes = extract(t.income_stmt, t.balance_sheet, t.cashflow,
                                   market_cap=info.get("marketCap", float("nan")))
            return {
                "company": info.get("longName") or info.get("shortName") or symbol,
                "ticker": symbol,
                "sector": info.get("sector"),
                **compute_metrics(items),
                "notes": ";".join(notes),
            }
        except Exception as exc:                      # noqa: BLE001 -- report, don't crash the run
            last_error = exc
            time.sleep(pause * (attempt + 1))         # back off; Yahoo rate-limits
    return {"company": symbol, "ticker": symbol, "sector": None,
            "roic": float("nan"), "fcf_yield": float("nan"),
            "rev_cagr_3y": float("nan"), "net_debt_ebitda": float("nan"),
            "notes": f"fetch failed: {type(last_error).__name__}: {last_error}"}
 
 
def fetch(symbols, pause: float = 1.0, raw_sectors: bool = False) -> pd.DataFrame:
    rows = []
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol}", file=sys.stderr)
        rows.append(line_items(symbol))
        if i < len(symbols):
            time.sleep(pause)
 
    df = pd.DataFrame(rows)
    if not raw_sectors:
        df["sector"] = df["sector"].map(lambda s: SECTOR_RENAME.get(s, s))
    df["as_of"] = date.today().isoformat()
    df["source"] = "Yahoo Finance via yfinance; metrics computed per README definitions"
    return df[["company", "ticker", "sector", "roic", "fcf_yield", "rev_cagr_3y",
               "net_debt_ebitda", "as_of", "source", "notes"]]
 
 
def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch screen inputs from Yahoo Finance.")
    ap.add_argument("tickers", nargs="*", help="ticker symbols")
    ap.add_argument("--tickers-file", help="file with one ticker per line")
    ap.add_argument("--out", default="data/companies.csv")
    ap.add_argument("--pause", type=float, default=1.0,
                    help="seconds between requests (Yahoo rate-limits; don't go below 0.5)")
    ap.add_argument("--raw-sectors", action="store_true",
                    help="keep Yahoo's sector names instead of GICS-style ones")
    args = ap.parse_args()
 
    symbols = list(args.tickers)
    if args.tickers_file:
        with open(args.tickers_file) as fh:
            symbols += [ln.strip().upper() for ln in fh
                        if ln.strip() and not ln.startswith("#")]
    if not symbols:
        ap.error("give tickers as arguments or via --tickers-file")
 
    df = fetch(symbols, pause=args.pause, raw_sectors=args.raw_sectors)
    df.to_csv(args.out, index=False)
 
    metrics = ["roic", "fcf_yield", "rev_cagr_3y", "net_debt_ebitda"]
    incomplete = df[df[metrics].isna().any(axis=1)]
    flagged = df[df["notes"].astype(bool)]
 
    print(f"\nwrote {len(df)} rows to {args.out}")
    if len(incomplete):
        print(f"\n{len(incomplete)} rows have at least one missing metric and will be "
              f"dropped by screen.py:")
        for _, r in incomplete.iterrows():
            print(f"  {r['ticker']:<8} {r['notes'] or 'see blank columns'}")
    if len(flagged):
        print(f"\n{len(flagged)} rows carry fetch notes -- read the notes column.")
    print("\nBefore ranking: spot-check ~10 names against their 10-K. Yahoo's "
          "statement data has gaps, and a wrong input ranks a company for no reason.")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())