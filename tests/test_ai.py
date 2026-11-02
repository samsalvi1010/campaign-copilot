"""Tests for src/ai.py. Run with:  pytest

No real API calls are made: ask_gemini is tested with a fake Gemini client,
so these tests run offline, for free, and always give the same result.
"""
import json

import pytest

import src.ai as ai
from src.ai import (ALLOWED_GROUPINGS, ALLOWED_METRICS, EXAMPLE_QUESTIONS, build_data_summary,
                    parse_ai_reply, rule_based_answer)
from src.kpis import load_campaign_data

CHANNELS = ["Email", "Google Ads", "LinkedIn", "Meta", "TikTok"]


@pytest.fixture(scope="module")
def sample_df():
    """The real sample dataset, loaded once for all tests in this file."""
    return load_campaign_data("data/sample_campaigns.csv")


def valid_reply(**changes):
    """A correct AI reply as a JSON string; pass changes to break it on purpose."""
    reply = {
        "answer": "Email has the best ROAS at 5.39x.",
        "chart": {"group_by": "channel", "metric": "roas", "channel": "All"},
        "recommendation": "Move €10,000 from TikTok to Email.",
    }
    reply.update(changes)
    return json.dumps(reply)


# --- Data summary ------------------------------------------------------------

def test_summary_has_all_sections_and_is_small(sample_df):
    summary = build_data_summary(sample_df)
    for section in ["By channel", "By country", "By campaign", "By month", "By week", "By channel and week"]:
        assert f"## {section}" in summary
    # The raw CSV is ~300 KB; the summary must be far smaller
    assert len(summary) < 15_000


def test_summary_does_not_contain_raw_rows(sample_df):
    summary = build_data_summary(sample_df)
    # Raw rows have a daily date like 2026-07-02; the summary only has week/month starts
    assert "2026-07-02" not in summary


# --- Checking the AI's JSON ---------------------------------------------------

def test_parse_valid_reply():
    reply = parse_ai_reply(valid_reply(), CHANNELS)
    assert reply["chart"] == {"group_by": "channel", "metric": "roas", "channel": "All"}
    assert reply["source"] == "ai"


def test_parse_rejects_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_ai_reply("Sure! Here is my answer...", CHANNELS)


def test_parse_rejects_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        parse_ai_reply(valid_reply(chart={"group_by": "channel", "metric": "profit", "channel": "All"}), CHANNELS)


def test_parse_rejects_missing_answer():
    with pytest.raises(ValueError, match="no answer"):
        parse_ai_reply(valid_reply(answer=""), CHANNELS)


def test_parse_unknown_channel_falls_back_to_all():
    reply = parse_ai_reply(valid_reply(chart={"group_by": "week", "metric": "roas", "channel": "Snapchat"}), CHANNELS)
    assert reply["chart"]["channel"] == "All"


# --- ask_gemini with a fake client ---------------------------------------------

class FakeModels:
    def __init__(self, reply_text):
        self.reply_text = reply_text
        self.last_call = None

    def generate_content(self, model, contents, config):
        self.last_call = {"model": model, "contents": contents, "config": config}

        class Response:
            text = self.reply_text
        return Response()


class FakeClient:
    def __init__(self, reply_text):
        self.models = FakeModels(reply_text)


def test_ask_gemini_sends_summary_and_parses_reply(monkeypatch):
    fake = FakeClient(valid_reply())
    monkeypatch.setattr(ai.genai, "Client", lambda **kwargs: fake)

    reply = ai.ask_gemini("Which channel is best?", "SUMMARY TEXT", CHANNELS, api_key="test-key")

    assert reply["answer"].startswith("Email")
    call = fake.models.last_call
    assert call["model"] == ai.MODEL_NAME
    assert "SUMMARY TEXT" in call["contents"] and "Which channel is best?" in call["contents"]
    assert call["config"].response_mime_type == "application/json"


def test_ask_gemini_raises_on_bad_reply(monkeypatch):
    monkeypatch.setattr(ai.genai, "Client", lambda **kwargs: FakeClient("not json"))
    with pytest.raises(ValueError):
        ai.ask_gemini("Anything?", "SUMMARY", CHANNELS, api_key="test-key")


# --- Rule-based backup ----------------------------------------------------------

@pytest.mark.parametrize("question", EXAMPLE_QUESTIONS + ["What's the weather like?"])
def test_rule_based_always_returns_valid_shape(sample_df, question):
    reply = rule_based_answer(question, sample_df)
    assert reply["answer"] and reply["recommendation"]
    assert reply["chart"]["group_by"] in ALLOWED_GROUPINGS
    assert reply["chart"]["metric"] in ALLOWED_METRICS
    assert reply["source"] == "rules"


def test_rule_based_finds_the_meta_drop(sample_df):
    reply = rule_based_answer("Why did Meta drop in September?", sample_df)
    assert "14 Sep 2026" in reply["answer"]
    assert reply["chart"] == {"group_by": "week", "metric": "roas", "channel": "Meta"}


def test_rule_based_budget_moves_from_tiktok(sample_df):
    reply = rule_based_answer("Where should I move 10% of budget?", sample_df)
    assert "from TikTok" in reply["recommendation"]


def test_rule_based_handles_single_channel(sample_df):
    only_meta = sample_df[sample_df["channel"] == "Meta"]
    reply = rule_based_answer("Where should I move 10% of budget?", only_meta)
    assert "Only one channel" in reply["answer"]


# --- Weekly report ----------------------------------------------------------------

def valid_report(**changes):
    """A correct AI report as a JSON string; pass changes to break it on purpose."""
    report = {
        "headline": "ROAS was 2.82x on €1,245,777 spend.",
        "wins": ["Win 1", "Win 2", "Win 3"],
        "issues": ["Issue 1", "Issue 2", "Issue 3"],
        "actions": ["Action 1", "Action 2", "Action 3"],
    }
    report.update(changes)
    return json.dumps(report)


def test_parse_valid_report():
    report = ai.parse_report_reply(valid_report())
    assert report["wins"] == ["Win 1", "Win 2", "Win 3"]
    assert report["source"] == "ai"


def test_parse_report_needs_exactly_three_items():
    with pytest.raises(ValueError, match="exactly 3 issues"):
        ai.parse_report_reply(valid_report(issues=["Only one"]))


def test_parse_report_rejects_empty_item():
    with pytest.raises(ValueError, match="empty item"):
        ai.parse_report_reply(valid_report(actions=["A", "  ", "C"]))


def test_generate_report_with_fake_client(monkeypatch):
    fake = FakeClient(valid_report())
    monkeypatch.setattr(ai.genai, "Client", lambda **kwargs: fake)
    report = ai.generate_report("SUMMARY TEXT", api_key="test-key")
    assert report["headline"].startswith("ROAS")
    assert "SUMMARY TEXT" in fake.models.last_call["contents"]


def test_rule_based_report_has_three_of_each(sample_df):
    report = ai.rule_based_report(sample_df)
    for key in ["wins", "issues", "actions"]:
        assert len(report[key]) == 3, key
    # The Meta drop should appear among the issues
    assert any("Meta" in issue and "14 Sep 2026" in issue for issue in report["issues"])


def test_rule_based_report_single_channel_does_not_crash(sample_df):
    report = ai.rule_based_report(sample_df[sample_df["channel"] == "Email"])
    assert report["headline"] and report["wins"] and report["actions"]


def test_report_markdown_has_sections_and_exact_numbers(sample_df):
    markdown = ai.report_to_markdown(ai.parse_report_reply(valid_report()), sample_df, "all data")
    for heading in ["# Campaign performance report", "## Key numbers", "## Wins", "## Issues",
                    "## Recommended actions"]:
        assert heading in markdown
    assert "1. Win 1" in markdown and "3. Action 3" in markdown
    # Key numbers come from pandas, not the AI
    assert "€1,245,777" in markdown and "2.82x" in markdown


# --- Edge cases found in code review ------------------------------------------------

def test_budget_text_when_amount_is_capped(sample_df):
    # LinkedIn barely spends anything, so "10% of total spend" is more than LinkedIn has:
    # the move is capped at LinkedIn's whole spend, and the text must say so
    df = sample_df[sample_df["channel"].isin(["Google Ads", "LinkedIn"])].copy()
    linkedin = df["channel"] == "LinkedIn"
    df.loc[linkedin, "spend_eur"] = 1.0
    df.loc[linkedin, "revenue_eur"] = 0.5  # ROAS 0.5x: the weakest channel
    reply = rule_based_answer("Where should I move budget?", df)
    assert "all of LinkedIn's spend" in reply["answer"]
    assert "10% of total spend" not in reply["answer"]


def test_summary_when_last_month_has_no_spend(sample_df):
    df = sample_df.copy()
    october = df["date"].dt.month == 10
    df.loc[october, ["spend_eur", "revenue_eur", "conversions"]] = 0
    reply = rule_based_answer("Summarise last month", df)  # must not crash
    assert "October 2026" in reply["answer"]


def test_drop_ignores_channel_with_zero_revenue(sample_df):
    df = sample_df[sample_df["channel"] == "TikTok"].copy()
    df["revenue_eur"] = 0  # ROAS is 0 every week: no "drop" to find
    reply = rule_based_answer("Why did TikTok drop?", df)
    assert "couldn't find a clear drop" in reply["answer"]


def test_chart_with_no_values_does_not_crash(sample_df):
    from src.charts import metric_chart
    from src.kpis import kpis_by
    table = kpis_by(sample_df, "channel")
    table["cpa"] = float("nan")  # e.g. no conversions anywhere
    fig = metric_chart(table, "channel", "cpa", "CPA by channel")
    assert "No data to show" in fig.layout.annotations[0].text
