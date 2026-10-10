"""Plotly charts for the Runway tab."""
import pandas as pd
import plotly.graph_objects as go


def _dates(as_of, horizon):
    start = pd.Timestamp(as_of) + pd.Timedelta(1, unit="D")
    return pd.date_range(start, periods=horizon, freq="D")


def fan_chart(sim, as_of, buffer=0.0):
    x = _dates(as_of, len(sim.p50))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=sim.p90, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=sim.p10, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(34,211,166,0.18)", name="Plausible range (P10 to P90)"))
    fig.add_trace(go.Scatter(x=x, y=sim.p90, mode="lines",
                             line=dict(width=1, dash="dot", color="#2ca02c"), name="Best case (P90)"))
    fig.add_trace(go.Scatter(x=x, y=sim.p10, mode="lines",
                             line=dict(width=1, dash="dot", color="#d62728"), name="Worst case (P10)"))
    fig.add_trace(go.Scatter(x=x, y=sim.p50, mode="lines",
                             line=dict(width=3, color="#22d3a6"), name="Likely case (P50)"))
    fig.add_hline(y=buffer, line_dash="dash", line_color="grey",
                  annotation_text="Zero cash" if not buffer else "Minimum buffer")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=420, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="Cash (PKR)", legend=dict(orientation="h", y=-0.2))
    return fig


def shortfall_chart(sim, as_of):
    x = _dates(as_of, len(sim.p_short_by_day))
    fig = go.Figure(go.Scatter(x=x, y=sim.p_short_by_day * 100, mode="lines", fill="tozeroy",
                               line=dict(color="#d62728"), name="Chance cash is below the buffer"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=300, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="Chance of shortfall on that day (%)", yaxis_range=[0, 100])
    return fig