"""Campaign Copilot: Streamlit entry point.

Run with:  streamlit run app.py

This file only handles the page: sidebar, KPI cards, charts, table, "Ask your data"
and the weekly report. The calculations live in src/kpis.py, the charts in
src/charts.py and the AI in src/ai.py.

How Streamlit works (useful to explain): every time the user clicks or types,
Streamlit re-runs this whole file from top to bottom. Anything we want to keep
between runs (like the last answer) is stored in st.session_state.
"""
import math
from pathlib import Path

import streamlit as st

from src.ai import (EXAMPLE_QUESTIONS, MODEL_NAME, ask_gemini, build_data_summary, generate_report,
                    report_to_markdown, rule_based_answer, rule_based_report)
from src.charts import (METRIC_INFO, cpa_by_channel_chart, metric_chart, roas_by_channel_chart,
                        weekly_spend_revenue_chart)
from src.kpis import REQUIRED_COLUMNS, calculate_kpis, kpis_by, load_campaign_data

SAMPLE_FILE = Path(__file__).parent / "data" / "sample_campaigns.csv"

# Must be the first Streamlit command. "wide" uses the full browser width.
st.set_page_config(page_title="Campaign Copilot", page_icon="📊", layout="wide")


# --- Helper functions -----------------------------------------------------

@st.cache_data(show_spinner="Loading sample data...")
def load_sample_data():
    """Load the sample CSV once and reuse it (cache) instead of re-reading on every click."""
    return load_campaign_data(SAMPLE_FILE)


def format_euro(value):
    """12345.6 -> '€12,346'. Shows 'n/a' when the value can't be calculated (NaN)."""
    if math.isnan(value):
        return "n/a"
    return f"€{value:,.0f}"


def format_euro_cents(value):
    """12.345 -> '€12.35'. Used for small amounts like CPA."""
    if math.isnan(value):
        return "n/a"
    return f"€{value:,.2f}"


def format_ratio(value):
    """2.3456 -> '2.35x' (used for ROAS)."""
    if math.isnan(value):
        return "n/a"
    return f"{value:.2f}x"


def format_percent(value):
    """0.0345 -> '3.45%'."""
    if math.isnan(value):
        return "n/a"
    return f"{value:.2%}"


def safe_markdown(text):
    """Streamlit treats text between two $ signs as a maths formula; escape them."""
    return text.replace("$", "\\$")


def get_api_key():
    """Return the Gemini key from .streamlit/secrets.toml, or None if it isn't set."""
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        # No secrets file, or no GEMINI_API_KEY inside it
        return None
    if not key or key == "paste-your-key-here":
        return None
    return key


@st.cache_data(show_spinner=False)
def ask_gemini_cached(question, summary, channels, api_key):
    """Same as ask_gemini, but remembers answers so asking twice doesn't cost twice."""
    return ask_gemini(question, summary, channels, api_key)


def get_answer(question, data):
    """Ask Gemini if possible; otherwise use the rule-based backup.

    Returns (reply, note): the reply dict, and a short note explaining
    why the backup was used (None when the AI answered).
    """
    api_key = get_api_key()
    if api_key is None:
        note = ("No Gemini API key found, so this is a simpler rule-based answer. "
                "Add your key to .streamlit/secrets.toml to use the AI.")
        return rule_based_answer(question, data), note

    try:
        channels = sorted(data["channel"].unique())
        summary = build_data_summary(data)  # small KPI tables, never the raw CSV
        return ask_gemini_cached(question, summary, channels, api_key), None
    except Exception as error:
        # Any problem (no internet, bad key, rate limit, invalid JSON): don't crash
        print(f"Gemini call failed: {error!r}")  # full details go to the terminal only
        note = (f"The AI service didn't give a usable answer ({type(error).__name__}), "
                "so this is a simpler rule-based answer. Please try again in a moment.")
        return rule_based_answer(question, data), note


@st.cache_data(show_spinner=False)
def generate_report_cached(summary, api_key):
    """Same as generate_report, but remembers reports for the same data."""
    return generate_report(summary, api_key)


def get_report(data):
    """Get the report from Gemini if possible, otherwise from the rule-based backup.

    Returns (report, note), just like get_answer().
    """
    api_key = get_api_key()
    if api_key is None:
        note = "No Gemini API key found, so this report was built with simple rules instead of AI."
        return rule_based_report(data), note
    try:
        return generate_report_cached(build_data_summary(data), api_key), None
    except Exception as error:
        print(f"Gemini report failed: {error!r}")
        note = (f"The AI service didn't give a usable report ({type(error).__name__}), "
                "so this one was built with simple rules. Please try again in a moment.")
        return rule_based_report(data), note


def show_reply(reply, data):
    """Display the answer, its chart and the recommendation in one card."""
    chart = reply["chart"]
    with st.container(border=True):
        st.markdown("##### Answer")
        st.markdown(safe_markdown(reply["answer"]))

        # Draw the chart the answer asked for, using our own chart functions
        chart_data = data if chart["channel"] == "All" else data[data["channel"] == chart["channel"]]
        if not chart_data.empty:
            title = f"{METRIC_INFO[chart['metric']]['label']} by {chart['group_by']}"
            if chart["channel"] != "All":
                title += f" ({chart['channel']})"
            table = kpis_by(chart_data, chart["group_by"])
            st.plotly_chart(metric_chart(table, chart["group_by"], chart["metric"], title), width="stretch")

        st.success(f"**Recommendation:** {safe_markdown(reply['recommendation'])}", icon="💡")

        if reply["source"] == "ai":
            st.caption(f"Answered by Gemini ({MODEL_NAME}) from a summary of the filtered data. "
                       "Check key numbers before acting.")
        else:
            st.caption("Rule-based answer, calculated directly with pandas.")


def use_example(question):
    """Button callback: put the example in the text box and ask it straight away."""
    st.session_state["question"] = question
    st.session_state["ask_now"] = True


# --- Header ---------------------------------------------------------------

st.title("Campaign Copilot")
st.caption("Upload your marketing campaign data to see what's working, what isn't, and where to spend next.")


# --- Sidebar ----------------------------------------------------------------
# Sidebar items appear in the order they're created. We reserve an empty box for
# the filters now, so the About box can sit below them but still show up even
# before any data is loaded (st.stop() below would otherwise hide it).

st.sidebar.header("Data")
use_sample = st.sidebar.checkbox("Use sample data", value=True)  # on by default for demos
uploaded_file = st.sidebar.file_uploader("Or upload your own CSV", type="csv", disabled=use_sample)

with st.sidebar.expander("Expected CSV format"):
    st.markdown("One row per day, campaign and country, with these columns:")
    st.code(", ".join(REQUIRED_COLUMNS), language=None)
    st.download_button("Download the sample CSV", data=SAMPLE_FILE.read_bytes(),
                       file_name="sample_campaigns.csv", mime="text/csv", on_click="ignore")

filters_box = st.sidebar.container()  # filled in later, once data is loaded

with st.sidebar.container(border=True):
    st.markdown(
        "**About**  \n"
        "Built by **Samarth Salvi** as a data & AI portfolio project.  \n"
        "[LinkedIn](https://www.linkedin.com/in/samarth-salvi-1010) · "
        "[GitHub](https://github.com/samsalvi1010)"
    )
    st.caption("The sample data is synthetic (generated for this demo).")


# --- Load the data ------------------------------------------------------------

try:
    if use_sample:
        df = load_sample_data()
    elif uploaded_file is not None:
        with st.spinner("Reading and cleaning your file..."):
            df = load_campaign_data(uploaded_file)
    else:
        st.info("👈 Upload a campaign CSV or tick **Use sample data** in the sidebar to get started.")
        st.stop()  # nothing to show yet, so stop running the rest of the page
except ValueError as error:
    # load_campaign_data raises ValueError with a friendly message (see src/kpis.py)
    st.error(f"**Could not load the file.** {error}")
    st.stop()


# --- Filters (inside the box reserved in the sidebar) ----------------------------

with filters_box:
    st.header("Filters")

    first_day = df["date"].min().date()
    last_day = df["date"].max().date()
    date_range = st.date_input("Date range", value=(first_day, last_day),
                               min_value=first_day, max_value=last_day)

    all_channels = sorted(df["channel"].unique())
    channels = st.multiselect("Channel", all_channels, default=all_channels)

    all_countries = sorted(df["country"].unique())
    countries = st.multiselect("Country", all_countries, default=all_countries)

# While the user is picking a range, date_input briefly returns only a start date
if len(date_range) != 2:
    st.info("Pick an end date to finish the date range.")
    st.stop()
start_date, end_date = date_range

# Keep only the rows that match every filter
filtered = df[
    (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
    & (df["channel"].isin(channels))
    & (df["country"].isin(countries))
]

if filtered.empty:
    st.warning("No data matches these filters. Try selecting more channels, countries or dates.")
    st.stop()

# A short description of the current filters (used by the report and to spot stale results)
filters_text = (
    f"{start_date:%d %b %Y} to {end_date:%d %b %Y}; "
    f"channels: {', '.join(channels) if len(channels) < len(all_channels) else 'all'}; "
    f"countries: {', '.join(countries) if len(countries) < len(all_countries) else 'all'}"
)


# --- Top row: KPI cards ---------------------------------------------------

kpis = calculate_kpis(filtered)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Spend", format_euro(kpis["spend_eur"]), border=True)
col2.metric("Revenue", format_euro(kpis["revenue_eur"]), border=True)
col3.metric("ROAS", format_ratio(kpis["roas"]), border=True,
            help="Revenue earned for every €1 spent. Below 1.00x means losing money.")
col4.metric("CPA", format_euro_cents(kpis["cpa"]), border=True,
            help="Cost per acquisition: average spend needed to get one conversion.")
col5.metric("Conversion rate", format_percent(kpis["conversion_rate"]), border=True,
            help="Share of clicks that became a conversion.")


# --- Charts ---------------------------------------------------------------

weekly = kpis_by(filtered, "week")
by_channel = kpis_by(filtered, "channel")

with st.container(border=True):
    st.plotly_chart(weekly_spend_revenue_chart(weekly), width="stretch")
    st.caption("Weeks run Monday to Sunday. The first and last week can be partial, so they may look lower.")

left, right = st.columns(2)
with left, st.container(border=True):
    st.plotly_chart(roas_by_channel_chart(by_channel), width="stretch")
with right, st.container(border=True):
    st.plotly_chart(cpa_by_channel_chart(by_channel), width="stretch")


# --- Campaign table -------------------------------------------------------

st.subheader("Campaigns ranked by ROAS")

campaigns = kpis_by(filtered, "campaign").sort_values("roas", ascending=False)
campaigns = campaigns[["campaign", "channel", "spend_eur", "revenue_eur", "conversions",
                       "roas", "cpa", "ctr", "conversion_rate"]]
# Rates are stored as fractions (0.04); multiply by 100 so the table shows "4.00%"
campaigns["ctr"] = campaigns["ctr"] * 100
campaigns["conversion_rate"] = campaigns["conversion_rate"] * 100

# column_config controls how each column is labelled and formatted on screen;
# the underlying numbers are unchanged
st.dataframe(
    campaigns,
    hide_index=True,
    width="stretch",
    column_config={
        "campaign": "Campaign",
        "channel": "Channel",
        "spend_eur": st.column_config.NumberColumn("Spend", format="€%,.0f"),
        "revenue_eur": st.column_config.NumberColumn("Revenue", format="€%,.0f"),
        "conversions": st.column_config.NumberColumn("Conversions", format="%,d"),
        "roas": st.column_config.NumberColumn("ROAS", format="%.2fx"),
        "cpa": st.column_config.NumberColumn("CPA", format="€%.2f"),
        "ctr": st.column_config.NumberColumn("CTR", format="%.2f%%"),
        "conversion_rate": st.column_config.NumberColumn("Conv. rate", format="%.2f%%"),
    },
)


# --- Ask your data ----------------------------------------------------------

st.divider()
st.header("Ask your data")
st.caption("Questions are answered from the filtered data above. "
           "Only a short summary of KPIs is sent to the AI, never your raw file.")

# Example question buttons, side by side
for column, example in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
    column.button(example, on_click=use_example, args=(example,), width="stretch")

# A form lets the user press Enter to submit
with st.form("ask_form", border=False):
    st.text_input("Your question", key="question", max_chars=300,
                  placeholder="e.g. Which country has the lowest CPA?")
    submitted = st.form_submit_button("Ask", type="primary")

# .pop() reads the flag set by an example button and clears it in one step
if submitted or st.session_state.pop("ask_now", False):
    question = st.session_state["question"].strip()
    if not question:
        st.warning("Please type a question first, or click one of the examples.")
    else:
        try:
            with st.spinner("Analysing your data... (the AI can take 10-30 seconds)"):
                reply, note = get_answer(question, filtered)
            # Save it, so the answer stays on screen when other buttons are clicked
            st.session_state["last_answer"] = {"question": question, "reply": reply,
                                               "note": note, "filters": filters_text}
        except Exception as error:
            # Last line of defence: even the rule-based backup failed
            print(f"Answering failed: {error!r}")
            st.error("Sorry, something went wrong while answering. "
                     "Try another question, or widen the filters.")

last_answer = st.session_state.get("last_answer")
if last_answer:
    st.markdown(f"**Q: {safe_markdown(last_answer['question'])}**")
    if last_answer["filters"] != filters_text:
        st.warning("The filters have changed since this question was answered. Ask again to update it.")
    show_reply(last_answer["reply"], filtered)
    if last_answer["note"]:
        st.info(last_answer["note"])


# --- Weekly report ------------------------------------------------------------

st.divider()
st.header("Weekly report")
st.caption("A short executive summary of the filtered data: 3 wins, 3 issues and 3 actions, "
           "ready to download as a Markdown file.")

if st.button("Generate weekly report", type="primary"):
    try:
        with st.spinner("Writing the report... (the AI can take 20-40 seconds)"):
            report, note = get_report(filtered)
        # Save in session_state so the report stays on screen after later clicks
        st.session_state["report"] = {
            "markdown": report_to_markdown(report, filtered, filters_text),
            "note": note,
            "filters": filters_text,
            "date": end_date,
        }
    except Exception as error:
        print(f"Report failed: {error!r}")
        st.error("Sorry, something went wrong while writing the report. Try widening the filters.")

saved = st.session_state.get("report")
if saved:
    if saved["filters"] != filters_text:
        st.warning("The filters have changed since this report was made. Click the button again to update it.")
    if saved["note"]:
        st.info(saved["note"])
    with st.container(border=True):
        st.markdown(safe_markdown(saved["markdown"]))
    st.download_button(
        "Download report (.md)",
        data=saved["markdown"],
        file_name=f"campaign_report_{saved['date']:%Y-%m-%d}.md",
        mime="text/markdown",
        on_click="ignore",  # downloading doesn't need to re-run the page
        icon=":material/download:",
    )
