"""Tests for src/kpis.py. Run with:  pytest"""
import io
import math

import pytest

from src.kpis import calculate_kpis, kpis_by, load_campaign_data, safe_divide

HEADER = "date,channel,campaign,country,impressions,clicks,spend_eur,conversions,revenue_eur\n"


def make_csv(rows):
    """Turn a few CSV lines into a file-like object, like an uploaded file."""
    return io.StringIO(HEADER + rows)


# A tiny dataset with easy numbers so we can check the maths by hand
SIMPLE_ROWS = (
    "2026-09-07,Meta,Prospecting,FR,1000,10,20,2,100\n"   # Monday
    "2026-09-08,Meta,Prospecting,DE,1000,30,40,2,60\n"    # Tuesday, same week
    "2026-09-14,TikTok,Spark Ads,FR,2000,40,10,0,0\n"     # Monday of the next week
)


# --- Loading and cleaning --------------------------------------------------

def test_load_parses_dates():
    df = load_campaign_data(make_csv(SIMPLE_ROWS))
    assert len(df) == 3
    assert str(df["date"].dtype).startswith("datetime64")


def test_load_real_sample_file():
    df = load_campaign_data("data/sample_campaigns.csv")
    assert len(df) > 0
    assert set(df["channel"]) == {"Google Ads", "Meta", "TikTok", "LinkedIn", "Email"}


def test_missing_column_gives_clear_error():
    csv = io.StringIO("date,channel,clicks\n2026-09-07,Meta,10\n")
    with pytest.raises(ValueError, match="Missing required column"):
        load_campaign_data(csv)


def test_empty_file_gives_clear_error():
    with pytest.raises(ValueError, match="empty"):
        load_campaign_data(io.StringIO(""))


def test_non_text_file_gives_clear_error():
    # Bytes that aren't valid UTF-8, like an Excel file renamed to .csv
    fake_excel = io.BytesIO(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1\xff\xfe")
    with pytest.raises(ValueError, match="plain-text CSV"):
        load_campaign_data(fake_excel)


def test_column_names_are_cleaned():
    csv = io.StringIO(
        " Date ,Channel,Campaign,Country,Impressions,Clicks,Spend_EUR,Conversions,Revenue_EUR\n"
        "2026-09-07,Meta,Prospecting,FR,1000,10,20,2,100\n"
    )
    df = load_campaign_data(csv)
    assert "spend_eur" in df.columns


def test_duplicates_are_dropped():
    row = "2026-09-07,Meta,Prospecting,FR,1000,10,20,2,100\n"
    df = load_campaign_data(make_csv(row + row))
    assert len(df) == 1


def test_missing_values_are_handled():
    rows = (
        "2026-09-07,,Prospecting,FR,1000,10,20,,100\n"       # blank channel and conversions
        "not-a-date,Meta,Prospecting,FR,1000,10,20,2,100\n"  # bad date -> row dropped
    )
    df = load_campaign_data(make_csv(rows))
    assert len(df) == 1
    assert df.loc[0, "channel"] == "Unknown"
    assert df.loc[0, "conversions"] == 0


def test_no_valid_rows_gives_clear_error():
    with pytest.raises(ValueError, match="No valid rows"):
        load_campaign_data(make_csv("not-a-date,Meta,Prospecting,FR,1000,10,20,2,100\n"))


# --- Division by zero ------------------------------------------------------

def test_safe_divide_normal_and_zero():
    assert safe_divide(10, 4) == 2.5
    assert math.isnan(safe_divide(10, 0))


# --- KPI calculations ------------------------------------------------------

def test_overall_kpis():
    kpis = calculate_kpis(load_campaign_data(make_csv(SIMPLE_ROWS)))
    # Totals: impressions 4000, clicks 80, spend 70, conversions 4, revenue 160
    assert kpis["ctr"] == pytest.approx(80 / 4000)
    assert kpis["cpc"] == pytest.approx(70 / 80)
    assert kpis["conversion_rate"] == pytest.approx(4 / 80)
    assert kpis["cpa"] == pytest.approx(70 / 4)
    assert kpis["roas"] == pytest.approx(160 / 70)


def test_kpis_by_channel():
    table = kpis_by(load_campaign_data(make_csv(SIMPLE_ROWS)), "channel").set_index("channel")
    # Meta: clicks 40, spend 60, conversions 4, revenue 160
    assert table.loc["Meta", "cpc"] == pytest.approx(60 / 40)
    assert table.loc["Meta", "roas"] == pytest.approx(160 / 60)
    # TikTok has 0 conversions: CPA must be NaN, not a crash and not 0
    assert math.isnan(table.loc["TikTok", "cpa"])


def test_kpis_by_week_groups_monday_to_sunday():
    table = kpis_by(load_campaign_data(make_csv(SIMPLE_ROWS)), "week")
    assert len(table) == 2  # 7 and 8 Sep share a week, 14 Sep is the next one
    assert list(table["week"].dt.strftime("%Y-%m-%d")) == ["2026-09-07", "2026-09-14"]
    assert list(table["clicks"]) == [40, 40]


def test_kpis_by_month():
    table = kpis_by(load_campaign_data(make_csv(SIMPLE_ROWS)), "month")
    assert len(table) == 1  # all three rows are in September
    assert table.loc[0, "month"].strftime("%Y-%m-%d") == "2026-09-01"


def test_kpis_by_country_and_campaign():
    df = load_campaign_data(make_csv(SIMPLE_ROWS))
    assert set(kpis_by(df, "country")["country"]) == {"FR", "DE"}
    assert set(kpis_by(df, "campaign")["campaign"]) == {"Prospecting", "Spark Ads"}


def test_invalid_group_gives_clear_error():
    df = load_campaign_data(make_csv(SIMPLE_ROWS))
    with pytest.raises(ValueError, match="Cannot group by"):
        kpis_by(df, "year")


def test_all_zero_data_does_not_crash():
    kpis = calculate_kpis(load_campaign_data(make_csv("2026-09-07,Meta,Prospecting,FR,0,0,0,0,0\n")))
    assert math.isnan(kpis["ctr"])
    assert math.isnan(kpis["roas"])
