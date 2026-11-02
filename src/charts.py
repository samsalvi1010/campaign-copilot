"""Functions that build plotly charts from KPI tables (made in src/kpis.py).

Each function takes a pandas DataFrame and returns a plotly Figure.
app.py then displays the figure with st.plotly_chart().

Design choices (worth mentioning in an interview):
  - One shared style (style_figure) so every chart looks consistent.
  - Spend and revenue share ONE euro axis. Two different y-axes on one chart can
    make lines look related when they aren't, so we avoid them.
  - Bar charts are sorted, so the best and worst channels are easy to spot.
"""
import plotly.graph_objects as go

# --- Colors ---------------------------------------------------------------
# A small, colorblind-friendly palette. Blue and orange are easy to tell apart
# even for people with the most common types of color blindness.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT_DARK = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def style_figure(fig, title):
    """Apply the shared clean look to any chart: white background, light grid, small margins."""
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": TEXT_DARK}, "x": 0, "xanchor": "left"},
        font={"family": FONT, "size": 13, "color": TEXT_MUTED},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 10, "r": 30, "t": 60, "b": 10},
        height=380,
        hoverlabel={"bgcolor": "white", "font_color": TEXT_DARK},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1},
    )
    # Thin, solid, light gridlines so the data stands out, not the grid
    fig.update_xaxes(showgrid=False, linecolor=AXIS, ticks="")
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def weekly_spend_revenue_chart(weekly):
    """Line chart of spend and revenue per week.

    `weekly` is the output of kpis_by(df, "week"): one row per week.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly["week"], y=weekly["spend_eur"], name="Spend",
        mode="lines+markers", line={"color": BLUE, "width": 2}, marker={"size": 6},
        hovertemplate="Spend: €%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=weekly["week"], y=weekly["revenue_eur"], name="Revenue",
        mode="lines+markers", line={"color": ORANGE, "width": 2}, marker={"size": 6},
        hovertemplate="Revenue: €%{y:,.0f}<extra></extra>",
    ))
    style_figure(fig, "Spend and revenue by week")
    # "x unified" shows both values in one tooltip when hovering over a week
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(tickformat="%d %b", hoverformat="Week of %d %b %Y")
    fig.update_yaxes(tickprefix="€", tickformat=",.0f", rangemode="tozero")
    return fig


# How to show each metric: a readable name, a number format, and whether
# a higher value is good (ROAS) or bad (CPA). Used by metric_chart below.
METRIC_INFO = {
    "spend_eur":       {"label": "Spend",           "format": "€{:,.0f}", "higher_is_better": True},
    "revenue_eur":     {"label": "Revenue",         "format": "€{:,.0f}", "higher_is_better": True},
    "conversions":     {"label": "Conversions",     "format": "{:,.0f}",  "higher_is_better": True},
    "roas":            {"label": "ROAS",            "format": "{:.2f}",   "higher_is_better": True},
    "cpa":             {"label": "CPA",             "format": "€{:,.2f}", "higher_is_better": False},
    "cpc":             {"label": "CPC",             "format": "€{:,.2f}", "higher_is_better": False},
    "ctr":             {"label": "CTR",             "format": "{:.2%}",   "higher_is_better": True},
    "conversion_rate": {"label": "Conversion rate", "format": "{:.2%}",   "higher_is_better": True},
}


def empty_chart(title):
    """A blank chart with a message, used when there's nothing to plot."""
    fig = go.Figure()
    style_figure(fig, title)
    fig.add_annotation(text="No data to show for the current filters", showarrow=False,
                       x=0.5, y=0.5, xref="paper", yref="paper", font={"color": TEXT_MUTED})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def horizontal_bar_chart(table, category_column, value_column, title, label_format, best_is_high):
    """Shared helper for "metric by category" bar charts (e.g. ROAS by channel).

    best_is_high=True puts the highest value at the top (good for ROAS);
    False puts the lowest at the top (good for CPA, where cheaper is better).
    """
    # Drop rows where the metric can't be calculated (e.g. CPA with 0 sales)
    table = table.dropna(subset=[value_column])
    if table.empty:
        return empty_chart(title)
    # Plotly draws horizontal bars bottom-to-top, so sort the opposite way
    table = table.sort_values(value_column, ascending=best_is_high)

    fig = go.Figure(go.Bar(
        x=table[value_column], y=table[category_column], orientation="h",
        marker={"color": BLUE, "cornerradius": 4},
        text=[label_format.format(v) for v in table[value_column]],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    style_figure(fig, title)
    fig.update_layout(bargap=0.45, showlegend=False)
    # For horizontal bars the grid is more useful on the x-axis.
    # Leave 20% extra room on the right so the value labels are never cut off.
    fig.update_xaxes(showgrid=True, gridcolor=GRID, range=[0, table[value_column].max() * 1.2])
    fig.update_yaxes(showgrid=False)
    return fig


def time_series_chart(table, time_column, value_column, title, label_format):
    """Single-line chart of one metric over time (weeks or months)."""
    if table[value_column].isna().all():
        return empty_chart(title)
    fig = go.Figure(go.Scatter(
        x=table[time_column], y=table[value_column],
        mode="lines+markers", line={"color": BLUE, "width": 2}, marker={"size": 7},
        text=[label_format.format(v) for v in table[value_column]],
        hovertemplate="%{x|%d %b %Y}: %{text}<extra></extra>",
    ))
    style_figure(fig, title)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickformat="%d %b" if time_column == "week" else "%b %Y")
    fig.update_yaxes(rangemode="tozero")
    return fig


def metric_chart(table, group_by, metric, title):
    """Draw any allowed metric for any allowed grouping (used by "Ask your data").

    Time groupings (week, month) become a line chart; categories become bars.
    `table` is the output of kpis_by(df, group_by).
    """
    info = METRIC_INFO[metric]
    if group_by in ("week", "month"):
        return time_series_chart(table, group_by, metric, title, info["format"])
    return horizontal_bar_chart(table, group_by, metric, title, info["format"],
                                best_is_high=info["higher_is_better"])


def roas_by_channel_chart(by_channel):
    """Bar chart of ROAS per channel, with a break-even line at 1.0.

    ROAS below 1 means a channel earns back less than it costs.
    """
    fig = horizontal_bar_chart(by_channel, "channel", "roas", "ROAS by channel (revenue per €1 spent)",
                               label_format="{:.2f}", best_is_high=True)
    fig.add_vline(x=1, line_width=1, line_color=TEXT_MUTED, layer="below",
                  annotation_text="Break-even", annotation_position="top",
                  annotation_font_color=TEXT_MUTED)
    return fig


def cpa_by_channel_chart(by_channel):
    """Bar chart of CPA (cost per conversion) per channel. Lower is better."""
    fig = horizontal_bar_chart(by_channel, "channel", "cpa", "CPA by channel (cost per conversion)",
                               label_format="€{:,.2f}", best_is_high=False)
    fig.update_xaxes(tickprefix="€")
    return fig
