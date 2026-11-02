"""The "Ask your data" brain: summarise the data, ask Gemini, check its reply.

How it works:
  1. build_data_summary() turns the filtered data into small KPI tables (text).
     We never send the raw CSV: it's smaller, cheaper, faster, and more private.
  2. ask_gemini() sends the summary + question and asks for a JSON reply.
  3. parse_ai_reply() checks the JSON has exactly what we expect.
  4. rule_based_answer() is a simple backup when there's no API key or the API fails.
  5. generate_report() / rule_based_report() do the same for the weekly report,
     and report_to_markdown() turns either one into a downloadable Markdown file.

Every answer (AI or rule-based) is a dictionary with the same shape:
    {
        "answer": "2-4 sentences with numbers",
        "chart": {"group_by": "channel", "metric": "roas", "channel": "All"},
        "recommendation": "one concrete action",
        "source": "ai" or "rules",
    }
"""
import json

import pandas as pd
from google import genai
from google.genai import types

from src.kpis import calculate_kpis, kpis_by

# Change the model here, in one place. See https://ai.google.dev/gemini-api/docs/models
# gemini-3.8-flash is newer but was timing out (504 errors) when this was built;
# switch back by changing this one line once it responds reliably.
MODEL_NAME = "gemini-3.6-flash"

# The only groupings and metrics the AI may ask us to chart.
# Limiting the choices means the AI can never request a chart we can't draw.
ALLOWED_GROUPINGS = ["channel", "campaign", "country", "week", "month"]
ALLOWED_METRICS = ["spend_eur", "revenue_eur", "conversions", "roas", "cpa", "cpc", "ctr", "conversion_rate"]

EXAMPLE_QUESTIONS = [
    "Which channel has the best ROAS?",
    "Why did Meta drop in September?",
    "Where should I move 10% of budget?",
    "Summarise last month in 3 bullets",
]

# Instructions the model sees before every question
SYSTEM_PROMPT = """You are a senior marketing analytics consultant.
You answer questions about a marketing campaign dataset using ONLY the data summary provided.
Currency is EUR. Columns ending in _pct are percentages. Weeks start on Monday, and the first
and last week may be partial weeks. An empty value means it could not be calculated.

Reply in JSON with:
- answer: 2-4 sentences that quote specific numbers from the summary. If the user asks for
  bullets, write up to 3 bullets, each starting with "- " and separated by a newline character.
  If the data cannot answer the question, say so honestly instead of guessing.
- chart: the single most useful chart for this answer.
  group_by is one of: channel, campaign, country, week, month.
  metric is one of: spend_eur, revenue_eur, conversions, roas, cpa, cpc, ctr, conversion_rate.
  channel is "All", or one channel name to show only that channel.
- recommendation: ONE concrete next action, with numbers where possible (e.g. euro amounts).

Number style: write money with the euro sign and thousands separators (€12,345), ROAS with
an x (2.36x), rates as percentages (3.45%), and dates like "14 Sep 2026"."""


# ---------------------------------------------------------------------------
# 1. Summarise the data (this is what gets sent instead of the raw CSV)
# ---------------------------------------------------------------------------

def table_to_text(table, label_columns):
    """Turn a KPI table into compact CSV text with sensible rounding."""
    out = table[label_columns].copy()

    # Dates as plain text, e.g. "2026-09-14"
    for col in label_columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")

    out["spend_eur"] = table["spend_eur"].round(0)
    out["revenue_eur"] = table["revenue_eur"].round(0)
    out["conversions"] = table["conversions"].round(0)
    out["roas"] = table["roas"].round(2)
    out["cpa"] = table["cpa"].round(2)
    out["cpc"] = table["cpc"].round(2)
    # Show rates as percentages so the model doesn't confuse 0.035 and 3.5
    out["ctr_pct"] = (table["ctr"] * 100).round(2)
    out["conversion_rate_pct"] = (table["conversion_rate"] * 100).round(2)
    return out.to_csv(index=False)


def build_data_summary(df):
    """Build a short text summary of the (filtered) data for the AI.

    It contains overall KPIs plus KPI tables by channel, country, campaign,
    month, week, and channel-by-week (needed to spot one channel dropping).
    """
    overall = calculate_kpis(df)
    start = df["date"].min().strftime("%Y-%m-%d")
    end = df["date"].max().strftime("%Y-%m-%d")

    parts = [
        f"Data from {start} to {end}.",
        (
            f"Overall: spend {overall['spend_eur']:,.0f}, revenue {overall['revenue_eur']:,.0f}, "
            f"ROAS {overall['roas']:.2f}, CPA {overall['cpa']:.2f}, "
            f"conversion rate {overall['conversion_rate'] * 100:.2f}%."
        ),
    ]

    sections = [
        ("By channel", "channel", ["channel"]),
        ("By country", "country", ["country"]),
        ("By campaign", "campaign", ["channel", "campaign"]),
        ("By month", "month", ["month"]),
        ("By week (all channels)", "week", ["week"]),
    ]
    for title, group_by, label_columns in sections:
        parts.append(f"\n## {title}\n" + table_to_text(kpis_by(df, group_by), label_columns))

    # Weekly numbers for each channel separately: without this, a problem in one
    # channel (like Meta in September) gets hidden inside the all-channel totals
    per_channel_weeks = []
    for channel in sorted(df["channel"].unique()):
        weekly = kpis_by(df[df["channel"] == channel], "week")
        weekly.insert(0, "channel", channel)
        per_channel_weeks.append(weekly)
    channel_weeks = pd.concat(per_channel_weeks, ignore_index=True)
    parts.append("\n## By channel and week\n" + table_to_text(channel_weeks, ["channel", "week"]))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 2. Ask Gemini
# ---------------------------------------------------------------------------

def build_response_schema(channels):
    """The JSON structure Gemini must follow. `enum` limits a field to a fixed list."""
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "chart": {
                "type": "object",
                "properties": {
                    "group_by": {"type": "string", "enum": ALLOWED_GROUPINGS},
                    "metric": {"type": "string", "enum": ALLOWED_METRICS},
                    "channel": {"type": "string", "enum": ["All"] + list(channels)},
                },
                "required": ["group_by", "metric", "channel"],
            },
            "recommendation": {"type": "string"},
        },
        "required": ["answer", "chart", "recommendation"],
    }


def call_gemini_json(contents, system_prompt, schema, api_key):
    """Send one request to Gemini and return its reply text (which should be JSON).

    Shared by "Ask your data" and the weekly report, so the settings live in one place.
    """
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30_000,  # milliseconds: give up after 30 seconds instead of hanging
            # If Google is busy (e.g. "503 high demand"), wait briefly and try once more
            retry_options=types.HttpRetryOptions(attempts=2),
        ),
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            # Ask for JSON that follows our schema instead of free text
            response_mime_type="application/json",
            response_json_schema=schema,
        ),
    )
    return response.text


def ask_gemini(question, summary, channels, api_key):
    """Send the question + data summary to Gemini and return a checked reply dict.

    Raises an exception if the API call fails or the reply is invalid;
    the app catches it and falls back to rule_based_answer().
    """
    text = call_gemini_json(
        contents=f"Data summary:\n{summary}\n\nQuestion: {question}",
        system_prompt=SYSTEM_PROMPT,
        schema=build_response_schema(channels),
        api_key=api_key,
    )
    return parse_ai_reply(text, channels)


def parse_ai_reply(text, channels):
    """Check the AI's JSON reply. Returns a clean dict, or raises ValueError.

    Even with a schema we double-check: never trust outside input blindly.
    """
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        raise ValueError("The AI reply was not valid JSON.")

    if not isinstance(data, dict):
        raise ValueError("The AI reply was not a JSON object.")

    answer = data.get("answer")
    recommendation = data.get("recommendation")
    chart = data.get("chart")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The AI reply has no answer.")
    if not isinstance(recommendation, str) or not recommendation.strip():
        raise ValueError("The AI reply has no recommendation.")
    if not isinstance(chart, dict):
        raise ValueError("The AI reply has no chart.")
    if chart.get("group_by") not in ALLOWED_GROUPINGS:
        raise ValueError(f"The AI asked for an unknown grouping: {chart.get('group_by')}")
    if chart.get("metric") not in ALLOWED_METRICS:
        raise ValueError(f"The AI asked for an unknown metric: {chart.get('metric')}")

    # An unknown channel isn't worth failing over: just show all channels
    channel = chart.get("channel")
    if channel not in channels:
        channel = "All"

    return {
        "answer": answer.strip(),
        "chart": {"group_by": chart["group_by"], "metric": chart["metric"], "channel": channel},
        "recommendation": recommendation.strip(),
        "source": "ai",
    }


# ---------------------------------------------------------------------------
# 3. Rule-based backup (no AI needed)
# ---------------------------------------------------------------------------

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def make_reply(answer, group_by, metric, channel, recommendation):
    """Build a reply dict in the same shape as the AI's, marked as rule-based."""
    return {
        "answer": answer,
        "chart": {"group_by": group_by, "metric": metric, "channel": channel},
        "recommendation": recommendation,
        "source": "rules",
    }


def rule_based_answer(question, df):
    """Answer common question types with simple pandas rules, using keywords.

    Much less flexible than the AI, but always works, even offline.
    """
    q = question.lower()

    # Channels ranked by ROAS, best first (channels with no spend are skipped)
    ranked = kpis_by(df, "channel").dropna(subset=["roas"]).sort_values("roas", ascending=False)
    if ranked.empty:
        return make_reply("There isn't enough spend data to answer this.", "channel", "spend_eur", "All",
                          "Widen the filters to include more data.")

    if any(word in q for word in ["budget", "move", "shift", "reallocat", "invest"]):
        return budget_answer(ranked)
    if any(word in q for word in ["why", "drop", "declin", "fell", "fall", "worse", "down"]):
        return drop_answer(q, df)
    if any(word in q for word in ["summar", "overview", "recap", "last month", "bullet"]):
        return summary_answer(df, ranked)
    return best_channel_answer(q, ranked)


def best_channel_answer(q, ranked):
    """Which channel has the best (and worst) ROAS."""
    best = ranked.iloc[0]
    worst = ranked.iloc[-1]

    answer = (
        f"{best['channel']} has the best ROAS at {best['roas']:.2f}x: every €1 spent "
        f"brought back €{best['roas']:.2f} in revenue. "
    )
    if len(ranked) > 1:
        answer += f"The weakest is {worst['channel']} at {worst['roas']:.2f}x."

    # If the question wasn't really about ROAS, be honest that this is a general answer
    if not any(word in q for word in ["best", "top", "roas", "highest", "worst"]):
        answer = "Without AI I can only answer common questions, so here is the overall picture. " + answer

    if worst["roas"] < 1:
        recommendation = (
            f"Review {worst['channel']}: at {worst['roas']:.2f}x ROAS it loses "
            f"€{1 - worst['roas']:.2f} for every €1 spent. Cap its budget until it reaches break-even."
        )
    else:
        recommendation = f"Test a 10-20% budget increase on {best['channel']} and check that its ROAS holds."
    return make_reply(answer, "channel", "roas", "All", recommendation)


def budget_answer(ranked):
    """Suggest moving 10% of total spend from the worst to the best ROAS channel."""
    if len(ranked) < 2:
        return make_reply("Only one channel is selected, so there's nowhere to move budget to.",
                          "channel", "roas", "All", "Select more channels in the sidebar filters.")

    best = ranked.iloc[0]
    worst = ranked.iloc[-1]
    total_spend = ranked["spend_eur"].sum()
    # Move 10% of total spend, but never more than the worst channel actually spends
    amount = min(0.10 * total_spend, worst["spend_eur"])
    if amount < 0.10 * total_spend:
        amount_text = f"all of {worst['channel']}'s spend"
    else:
        amount_text = "10% of total spend"
    # Rough estimate: assumes each channel keeps its current ROAS
    extra_revenue = amount * (best["roas"] - worst["roas"])

    answer = (
        f"{worst['channel']} has the lowest ROAS ({worst['roas']:.2f}x) and {best['channel']} "
        f"the highest ({best['roas']:.2f}x). Moving €{amount:,.0f} ({amount_text}) from "
        f"{worst['channel']} to {best['channel']} could add about €{extra_revenue:,.0f} in revenue, "
        f"if {best['channel']}'s ROAS holds at the higher budget."
    )
    recommendation = (
        f"Move €{amount:,.0f} from {worst['channel']} to {best['channel']} as a 2-week test, "
        f"and check {best['channel']}'s ROAS every week before moving more."
    )
    return make_reply(answer, "channel", "roas", "All", recommendation)


def find_biggest_drop(df, channels, month=None):
    """Find the channel-week whose ROAS is furthest below that channel's typical week.

    Returns a dict (channel, week row, typical values, drop as a fraction),
    or None if there's no data. Optionally only looks at weeks in one month.
    """
    worst = None  # we'll keep the biggest drop found so far
    for channel in channels:
        weekly = kpis_by(df[df["channel"] == channel], "week").dropna(subset=["roas"])
        if weekly.empty:
            continue
        typical = weekly.median(numeric_only=True)  # a "normal" week for this channel
        if typical["roas"] == 0:
            continue  # a channel that never earns revenue has no "drop" to measure
        candidates = weekly if month is None else weekly[weekly["week"].dt.month == month]
        for _, week in candidates.iterrows():
            drop = 1 - week["roas"] / typical["roas"]
            if worst is None or drop > worst["drop"]:
                worst = {"channel": channel, "week": week, "typical": typical, "drop": drop}
    return worst


def drop_answer(q, df):
    """Find the channel-week with the biggest ROAS drop and explain what changed."""
    # If the question names a channel (e.g. "Meta"), only look at that channel
    channels = [c for c in df["channel"].unique() if c.lower() in q] or list(df["channel"].unique())
    # If it names a month (e.g. "September"), only look for drops in that month
    month = next((number for name, number in MONTHS.items() if name in q), None)

    worst = find_biggest_drop(df, channels, month)
    if worst is None or worst["drop"] <= 0.10:
        return make_reply("I couldn't find a clear drop (more than 10% below a typical week) in the selected data.",
                          "week", "roas", channels[0] if len(channels) == 1 else "All",
                          "Check the weekly ROAS trend per channel using the filters.")

    channel, week, typical = worst["channel"], worst["week"], worst["typical"]
    week_label = week["week"].strftime("%d %b %Y")

    # Work out which parts of the funnel got worse (by more than 20%)
    reasons = []
    if week["cpc"] > typical["cpc"] * 1.2:
        reasons.append(f"clicks cost more (CPC €{week['cpc']:.2f} vs €{typical['cpc']:.2f})")
    if week["conversion_rate"] < typical["conversion_rate"] * 0.8:
        reasons.append(f"fewer clicks converted ({week['conversion_rate']:.2%} vs {typical['conversion_rate']:.2%})")
    if week["ctr"] < typical["ctr"] * 0.8:
        reasons.append(f"fewer people clicked (CTR {week['ctr']:.2%} vs {typical['ctr']:.2%})")
    reason_text = " and ".join(reasons) if reasons else "no single funnel step stands out"

    answer = (
        f"{channel}'s ROAS fell to {week['roas']:.2f}x in the week of {week_label}, "
        f"{worst['drop']:.0%} below its typical {typical['roas']:.2f}x. "
        f"In that week {reason_text}. "
        f"The data shows what changed, but not the cause (e.g. a new creative, audience or bid change)."
    )
    recommendation = (
        f"Check {channel}'s change history for the week of {week_label}, and set a weekly alert "
        f"if ROAS falls below {typical['roas'] * 0.8:.2f}x (20% under normal)."
    )
    return make_reply(answer, "week", "roas", channel, recommendation)


def summary_answer(df, ranked):
    """Three bullets about the most recent month in the data."""
    monthly = kpis_by(df, "month")
    last = monthly.iloc[-1]
    month_name = last["month"].strftime("%B %Y")
    month_data = df[df["date"].dt.to_period("M") == last["month"].to_period("M")]

    line1 = f"- {month_name}: spend €{last['spend_eur']:,.0f}, revenue €{last['revenue_eur']:,.0f}, ROAS {last['roas']:.2f}x"
    if len(monthly) > 1:
        previous = monthly.iloc[-2]
        line1 += f" (vs {previous['roas']:.2f}x the month before)"

    month_channels = kpis_by(month_data, "channel").dropna(subset=["roas"]).sort_values("roas", ascending=False)
    if month_channels.empty:
        # e.g. the last month has no spend at all: fall back to the whole period
        month_channels = ranked
    best, worst = month_channels.iloc[0], month_channels.iloc[-1]
    line2 = f"- Best channel: {best['channel']} ({best['roas']:.2f}x ROAS); weakest: {worst['channel']} ({worst['roas']:.2f}x)"

    campaigns = kpis_by(month_data, "campaign").dropna(subset=["roas"]).sort_values("roas", ascending=False)
    if campaigns.empty:
        line3 = "- No campaign had spend in this month, so campaigns can't be ranked"
    else:
        top = campaigns.iloc[0]
        line3 = f"- Top campaign: {top['campaign']} ({top['channel']}) at {top['roas']:.2f}x ROAS"

    if worst["roas"] < 1:
        recommendation = (
            f"Fix or cut {worst['channel']}: it lost money in {month_name} "
            f"(€{worst['spend_eur']:,.0f} spent for €{worst['revenue_eur']:,.0f} revenue)."
        )
    else:
        recommendation = f"Test a 10-20% budget increase on {best['channel']}, the best performer in {month_name}."
    return make_reply("\n".join([line1, line2, line3]), "week", "revenue_eur", "All", recommendation)


# ---------------------------------------------------------------------------
# 4. Weekly report
# ---------------------------------------------------------------------------

REPORT_PROMPT = """You are a senior marketing analytics consultant writing a short executive
summary for a marketing director, using ONLY the data summary provided.
Currency is EUR. Columns ending in _pct are percentages. Weeks start on Monday, and the first
and last week may be partial weeks (don't call a partial week a drop).

Reply in JSON with:
- headline: one sentence on overall performance, with numbers.
- wins: exactly 3 things that went well.
- issues: exactly 3 problems or risks.
- actions: exactly 3 concrete recommended actions (who/what/how much), tied to the issues and wins.
Each item is one sentence that includes specific numbers from the summary. Cover the whole
period, and mention how the most recent full week compares with the week before.

Number style: euro sign and thousands separators (€12,345), ROAS with an x (2.36x),
percentages (3.45%), dates like "14 Sep 2026". Plain text only, no markdown."""

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "wins": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
        "issues": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
        "actions": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
    },
    "required": ["headline", "wins", "issues", "actions"],
}


def generate_report(summary, api_key):
    """Ask Gemini for the executive summary. Returns a checked report dict, or raises."""
    text = call_gemini_json(
        contents=f"Data summary:\n{summary}\n\nWrite the executive summary.",
        system_prompt=REPORT_PROMPT,
        schema=REPORT_SCHEMA,
        api_key=api_key,
    )
    return parse_report_reply(text)


def parse_report_reply(text):
    """Check the report JSON: a headline plus 3 wins, 3 issues and 3 actions."""
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        raise ValueError("The AI report was not valid JSON.")
    if not isinstance(data, dict):
        raise ValueError("The AI report was not a JSON object.")

    headline = data.get("headline")
    if not isinstance(headline, str) or not headline.strip():
        raise ValueError("The AI report has no headline.")

    report = {"headline": headline.strip(), "source": "ai"}
    for key in ["wins", "issues", "actions"]:
        items = data.get(key)
        # Must be a list of 3 non-empty pieces of text
        if not isinstance(items, list) or len(items) != 3:
            raise ValueError(f"The AI report needs exactly 3 {key}.")
        if not all(isinstance(item, str) and item.strip() for item in items):
            raise ValueError(f"The AI report has an empty item in {key}.")
        report[key] = [item.strip() for item in items]
    return report


def rule_based_report(df):
    """Backup report built with pandas only. May have fewer than 3 items if data is thin."""
    overall = calculate_kpis(df)
    ranked = kpis_by(df, "channel").dropna(subset=["roas"]).sort_values("roas", ascending=False)
    campaigns = kpis_by(df, "campaign").dropna(subset=["roas"]).sort_values("roas", ascending=False)
    weekly = kpis_by(df, "week")

    headline = (
        f"Spend of €{overall['spend_eur']:,.0f} generated €{overall['revenue_eur']:,.0f} in revenue "
        f"(ROAS {overall['roas']:.2f}x, CPA €{overall['cpa']:.2f})."
    )
    wins, issues, actions = [], [], []

    if not ranked.empty:
        best, worst = ranked.iloc[0], ranked.iloc[-1]
        wins.append(f"{best['channel']} is the most efficient channel at {best['roas']:.2f}x ROAS "
                    f"(€{best['revenue_eur']:,.0f} revenue on €{best['spend_eur']:,.0f} spend).")
        if len(ranked) > 1:
            issues.append(f"{worst['channel']} has the lowest ROAS at {worst['roas']:.2f}x "
                          f"(€{worst['spend_eur']:,.0f} spend for €{worst['revenue_eur']:,.0f} revenue).")
            actions.append(budget_answer(ranked)["recommendation"])

    # Biggest one-week ROAS drop in any channel (an important issue, so it goes early)
    drop = find_biggest_drop(df, list(df["channel"].unique()))
    if drop is not None and drop["drop"] > 0.10:
        issues.append(f"{drop['channel']}'s ROAS fell to {drop['week']['roas']:.2f}x in the week of "
                      f"{drop['week']['week']:%d %b %Y}, {drop['drop']:.0%} below its typical "
                      f"{drop['typical']['roas']:.2f}x.")
        actions.append(f"Investigate what changed in {drop['channel']} in the week of "
                       f"{drop['week']['week']:%d %b %Y} and set a weekly ROAS alert.")

    if not campaigns.empty:
        top = campaigns.iloc[0]
        wins.append(f"Top campaign: {top['campaign']} ({top['channel']}) at {top['roas']:.2f}x ROAS.")

    # Compare the two most recent FULL weeks (7 days of data). A partial week
    # would look like a fake drop just because it has fewer days.
    days_per_week = df.groupby(df["date"].dt.to_period("W").dt.start_time)["date"].nunique()
    full_weeks = weekly[weekly["week"].isin(days_per_week[days_per_week == 7].index)]
    if len(full_weeks) >= 2:
        last, previous = full_weeks.iloc[-1], full_weeks.iloc[-2]
        change = last["revenue_eur"] / previous["revenue_eur"] - 1 if previous["revenue_eur"] else 0
        sentence = (f"Latest full week ({last['week']:%d %b %Y}): revenue €{last['revenue_eur']:,.0f}, "
                    f"{change:+.1%} vs the week before, at {last['roas']:.2f}x ROAS.")
        (wins if change >= 0 else issues).append(sentence)

    # Best country by ROAS
    by_country = kpis_by(df, "country").dropna(subset=["roas"]).sort_values("roas", ascending=False)
    if len(by_country) > 1:
        country = by_country.iloc[0]
        wins.append(f"{country['country']} is the strongest market at {country['roas']:.2f}x ROAS "
                    f"(€{country['revenue_eur']:,.0f} revenue).")

    # Highest cost per acquisition
    by_cpa = kpis_by(df, "channel").dropna(subset=["cpa"]).sort_values("cpa", ascending=False)
    if len(by_cpa) > 1:
        priciest = by_cpa.iloc[0]
        issues.append(f"{priciest['channel']} has the highest CPA at €{priciest['cpa']:.2f} per conversion.")

    actions.append("Review this report weekly and compare each channel's ROAS with the week before.")

    return {"headline": headline, "wins": wins[:3], "issues": issues[:3], "actions": actions[:3],
            "source": "rules"}


def report_to_markdown(report, df, filters_text):
    """Turn a report dict into a Markdown document.

    Key numbers come straight from pandas (always exact); the AI only writes the text.
    """
    overall = calculate_kpis(df)
    start = df["date"].min().strftime("%d %b %Y")
    end = df["date"].max().strftime("%d %b %Y")
    today = pd.Timestamp.today().strftime("%d %b %Y")
    written_by = f"Gemini ({MODEL_NAME})" if report["source"] == "ai" else "rule-based analysis (no AI)"

    lines = [
        "# Campaign performance report",
        "",
        f"**Period:** {start} to {end}  ",
        f"**Filters:** {filters_text}  ",
        f"**Generated:** {today} by Campaign Copilot, text written by {written_by}",
        "",
        f"> {report['headline']}",
        "",
        "## Key numbers",
        "",
        "| Spend | Revenue | ROAS | CPA | Conversion rate |",
        "|---:|---:|---:|---:|---:|",
        f"| €{overall['spend_eur']:,.0f} | €{overall['revenue_eur']:,.0f} | {overall['roas']:.2f}x "
        f"| €{overall['cpa']:,.2f} | {overall['conversion_rate']:.2%} |",
    ]
    for title, key in [("Wins", "wins"), ("Issues", "issues"), ("Recommended actions", "actions")]:
        lines += ["", f"## {title}", ""]
        lines += [f"{number}. {item}" for number, item in enumerate(report[key], start=1)]

    lines += ["", "---", "*Figures in the key numbers table are calculated directly from the data. "
              "Please double-check AI-written numbers before sharing.*", ""]
    return "\n".join(lines)
