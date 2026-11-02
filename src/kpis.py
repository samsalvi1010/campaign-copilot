"""Load and clean campaign data, then calculate marketing KPIs.

KPIs used in this project (rates are stored as fractions, e.g. 0.035 = 3.5%):
    CTR             = clicks / impressions        (how often people click the ad)
    CPC             = spend / clicks              (cost of one click)
    Conversion rate = conversions / clicks        (how often a click becomes a sale)
    CPA             = spend / conversions         (cost of one sale)
    ROAS            = revenue / spend             (revenue earned per 1 EUR spent)
"""
import pandas as pd

# Columns the CSV must contain, split by type so we know how to clean each one
TEXT_COLUMNS = ["channel", "campaign", "country"]
NUMBER_COLUMNS = ["impressions", "clicks", "spend_eur", "conversions", "revenue_eur"]
REQUIRED_COLUMNS = ["date"] + TEXT_COLUMNS + NUMBER_COLUMNS

# The ways a user is allowed to group the KPIs
GROUP_OPTIONS = ["channel", "campaign", "country", "week", "month"]


# ---------------------------------------------------------------------------
# 1. Loading and cleaning
# ---------------------------------------------------------------------------

def load_campaign_data(file):
    """Read a campaign CSV and return a clean DataFrame.

    `file` can be a file path or an uploaded file object (like Streamlit's uploader).
    Raises ValueError with a clear, human-readable message if the file can't be used.
    """
    # Read the CSV, turning pandas' technical errors into friendly ones
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty. Please upload a CSV with campaign data.")
    except pd.errors.ParserError:
        raise ValueError("The file could not be read as a CSV. Please check its format.")
    except UnicodeDecodeError:
        # Happens with e.g. an Excel file renamed to .csv, or an unusual text encoding
        raise ValueError("The file isn't a plain-text CSV (UTF-8). If it's an Excel file, "
                         "use 'Save as > CSV UTF-8' first.")

    # Make column names consistent: " Spend_EUR " -> "spend_eur"
    df.columns = df.columns.str.strip().str.lower()

    # Stop early if any required column is missing, and say which ones
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}."
        )

    # Keep only the columns we need, in a fixed order
    df = df[REQUIRED_COLUMNS].copy()

    # Dates: anything that isn't a valid date becomes NaT ("not a time"),
    # and we drop those rows because we can't place them on a timeline
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Text columns: fill blanks with "Unknown" so the row still counts in totals
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # Number columns: non-numbers become NaN, then missing numbers become 0
    # (e.g. a blank "conversions" cell most likely means there were none)
    for col in NUMBER_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Remove exact duplicate rows (e.g. the same export pasted twice)
    df = df.drop_duplicates().reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows left after cleaning. Please check the dates in your file.")

    return df


# ---------------------------------------------------------------------------
# 2. KPI calculations
# ---------------------------------------------------------------------------

def safe_divide(numerator, denominator):
    """Divide without crashing when the denominator is 0.

    Returns NaN ("not a number") instead of an error. We use NaN rather than 0
    on purpose: a CPA of 0 would look like sales were free, which is misleading.
    Works for single numbers and for whole pandas columns.
    """
    if isinstance(denominator, pd.Series):
        # Replace 0 with NaN in the column; anything divided by NaN is NaN
        return numerator / denominator.replace(0, float("nan"))
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def add_kpi_columns(totals):
    """Add CTR, CPC, conversion rate, CPA and ROAS columns to a table of totals."""
    totals = totals.copy()
    totals["ctr"] = safe_divide(totals["clicks"], totals["impressions"])
    totals["cpc"] = safe_divide(totals["spend_eur"], totals["clicks"])
    totals["conversion_rate"] = safe_divide(totals["conversions"], totals["clicks"])
    totals["cpa"] = safe_divide(totals["spend_eur"], totals["conversions"])
    totals["roas"] = safe_divide(totals["revenue_eur"], totals["spend_eur"])
    return totals


def calculate_kpis(df):
    """Return overall totals and KPIs for the whole dataset as a dictionary.

    Important: KPIs are calculated from the summed totals, NOT by averaging
    each row's KPI. Averaging ratios would give small rows too much weight.
    """
    totals = df[NUMBER_COLUMNS].sum()

    return {
        "impressions": totals["impressions"],
        "clicks": totals["clicks"],
        "spend_eur": totals["spend_eur"],
        "conversions": totals["conversions"],
        "revenue_eur": totals["revenue_eur"],
        "ctr": safe_divide(totals["clicks"], totals["impressions"]),
        "cpc": safe_divide(totals["spend_eur"], totals["clicks"]),
        "conversion_rate": safe_divide(totals["conversions"], totals["clicks"]),
        "cpa": safe_divide(totals["spend_eur"], totals["conversions"]),
        "roas": safe_divide(totals["revenue_eur"], totals["spend_eur"]),
    }


def kpis_by(df, group_by):
    """Return a table of totals and KPIs, one row per group.

    group_by must be one of: "channel", "campaign", "country", "week", "month".
    For "week", each week is labelled by its Monday date;
    for "month", each month by its first day.
    """
    if group_by not in GROUP_OPTIONS:
        raise ValueError(f"Cannot group by '{group_by}'. Choose one of: {', '.join(GROUP_OPTIONS)}.")

    df = df.copy()
    group_columns = [group_by]

    if group_by == "week":
        # to_period("W") puts each date in a Monday-Sunday week;
        # start_time gives us that week's Monday as a readable label
        df["week"] = df["date"].dt.to_period("W").dt.start_time

    if group_by == "month":
        df["month"] = df["date"].dt.to_period("M").dt.start_time

    if group_by == "campaign":
        # Group by channel too: two channels might use the same campaign name
        # (e.g. "Retargeting"), and we don't want to mix them together
        group_columns = ["channel", "campaign"]

    # Sum the raw numbers per group first, then calculate KPIs from those sums
    totals = df.groupby(group_columns)[NUMBER_COLUMNS].sum().reset_index()
    return add_kpi_columns(totals)
