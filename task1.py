
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import os

# COLOURS
BLUE   = "#185FA5"
PINK   = "#E31668"
PURPLE = "#740653"
RED    = "#DA0C0C"
GREEN  = "#3B6D11"
GREY   = "#888780"
L_BLUE = "#E6F1FB"
L_PINK = "#F0D5E1"
L_PURPLE = "#E9B2D5"
L_RED  = "#FCEBEB"
BG     = "#F8F8F6"
CARD   = "#FFFFFF"
MUTED  = "#5F5E5A"

PALETTE = [BLUE, PINK, PURPLE, GREEN, RED, GREY,
           "#533AB7", "#D85A30", "#378ADD", "#639922"]

#Loading and cleaning data
CSV = r"C:\Users\BOITUMELO\OneDrive\Desktop\Data scientist work\data.csv"

def load_data(path=CSV):
    df = pd.read_csv(
        path,
        encoding="ISO-8859-1",
        dtype={"CustomerID": str},
        parse_dates=["InvoiceDate"],
    )
    df.dropna(subset=["Description", "StockCode"], inplace=True)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    NON_PROD = {"POST", "D", "M", "BANK CHARGES", "PADS", "DOT"}
    df = df[~df["StockCode"].astype(str).str.upper().isin(NON_PROD)]
    df["Revenue"]    = df["Quantity"] * df["UnitPrice"]
    df["Year"]       = df["InvoiceDate"].dt.year
    df["Month"]      = df["InvoiceDate"].dt.month
    df["MonthLabel"] = df["InvoiceDate"].dt.strftime("%b %Y")
    df["YearMonth"]  = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayOfWeek"]  = df["InvoiceDate"].dt.day_name()
    df["Description"] = df["Description"].str.title().str.strip()

    CATS = {
        "Gifts & Novelties": ["gift","novelty","charm","heart","love","cute","bunting","balloon"],
        "Home Decor":        ["holder","light","lantern","frame","mirror","clock","cushion","vase","candle"],
        "Kitchen":           ["jar","tin","mug","bag","tray","bowl","bottle","storage","spoon"],
        "Stationery":        ["card","notebook","pen","pencil","paper","tag","label","wrap"],
        "Seasonal":          ["christmas","easter","halloween","santa","xmas"],
    }
    def categorise(d):
        d = str(d).lower()
        for cat, kws in CATS.items():
            if any(k in d for k in kws): return cat
        return "Other"
    df["Category"] = df["Description"].apply(categorise)
    return df


def build_rfm(df):
    snap = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm  = (df.dropna(subset=["CustomerID"])
              .groupby("CustomerID")
              .agg(Recency  =("InvoiceDate", lambda x: (snap - x.max()).days),
                   Frequency=("InvoiceNo",   "nunique"),
                   Monetary =("Revenue",     "sum"))
              .reset_index())
    rfm["R"] = pd.qcut(rfm["Recency"],   4, labels=[4,3,2,1]).astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"].rank(method="first"),  4, labels=[1,2,3,4]).astype(int)
    rfm["Score"] = rfm["R"] + rfm["F"] + rfm["M"]
    def seg(r):
        if r["Score"] >= 10:                       return "Champions"
        if r["Score"] >= 7:                        return "Loyal Customers"
        if r["R"] >= 3:                            return "Growth Potential"
        if r["R"] <= 2 and r["Score"] >= 6:        return "At Risk"
        return "Lost"
    rfm["Segment"] = rfm.apply(seg, axis=1)
    return rfm

#Economic layout
def kpi_card(title, value, delta=None, delta_pos=True, bg=CARD):
    delta_el = []
    if delta:
        colour = GREEN if delta_pos else RED
        arrow  = "▲" if delta_pos else "▼"
        delta_el = [html.P(f"{arrow} {delta}",
                           style={"margin":"4px 0 0","fontSize":"12px","color":colour,"fontWeight":"500"})]
    return html.Div([
        html.P(title, style={"margin":"0 0 4px","fontSize":"11px","color":MUTED,
                              "textTransform":"uppercase","letterSpacing":"0.06em"}),
        html.P(value, style={"margin":"0","fontSize":"24px","fontWeight":"500","color":"#2C2C2A"}),
        *delta_el,
    ], style={
        "background":bg, "borderRadius":"10px", "padding":"16px 20px",
        "border":"0.5px solid #D3D1C7", "minWidth":"140px", "flex":"1",
    })


def section(title, children, style_extra=None):
    base = {"background":CARD,"borderRadius":"12px","padding":"20px 24px",
            "border":"0.5px solid #D3D1C7","marginBottom":"16px"}
    if style_extra:
        base.update(style_extra)
    return html.Div([
        html.P(title, style={"margin":"0 0 14px","fontSize":"11px","fontWeight":"500",
                              "color":MUTED,"textTransform":"uppercase","letterSpacing":"0.07em"}),
        *children,
    ], style=base)


def fig_layout(fig, height=280):
    fig.update_layout(
        height=height, margin=dict(l=8,r=8,t=8,b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=MUTED, size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11)),
        xaxis=dict(gridcolor="#E8E6DF", linecolor="#D3D1C7"),
        yaxis=dict(gridcolor="#E8E6DF", linecolor="#D3D1C7"),
    )
    return fig

def fig_revenue_trend(df):
    monthly = (df.groupby(["Year","Month"])["Revenue"].sum().reset_index()
                 .sort_values(["Year","Month"]))
    fig = go.Figure()
    colors = {2010: BLUE, 2011: PINK}
    for year, grp in monthly.groupby("Year"):
        labels = pd.to_datetime(grp[["Year","Month"]].assign(day=1)).dt.strftime("%b")
        fig.add_trace(go.Scatter(
            x=labels, y=grp["Revenue"],
            mode="lines+markers", name=str(year),
            line=dict(color=colors.get(year, GREY), width=2.5),
            marker=dict(size=5),
            fill="tozeroy",
            fillcolor=f"rgba{tuple(int(colors.get(year,GREY).lstrip('#')[i:i+2],16) for i in (0,2,4))+(0.07,)}",
            hovertemplate="£%{y:,.0f}<extra>" + str(year) + "</extra>",
        ))
    fig_layout(fig, height=240)
    fig.update_yaxes(tickprefix="£", tickformat=",")
    return fig


def fig_top_products(df, n=10):
    top = (df.groupby("Description")["Revenue"].sum()
             .sort_values(ascending=True).tail(n).reset_index())
    fig = go.Figure(go.Bar(
        x=top["Revenue"], y=top["Description"],
        orientation="h",
        marker=dict(color=BLUE, line=dict(width=0)),
        hovertemplate="£%{x:,.0f}<extra></extra>",
    ))
    fig_layout(fig, height=max(300, n*34))
    fig.update_xaxes(tickprefix="£", tickformat=",")
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


def fig_country_revenue(df, n=10):
    top = (df.groupby("Country")["Revenue"].sum()
             .sort_values(ascending=True).tail(n).reset_index())
    fig = go.Figure(go.Bar(
        x=top["Revenue"], y=top["Country"],
        orientation="h",
        marker=dict(color=PINK, line=dict(width=0)),
        hovertemplate="£%{x:,.0f}<extra></extra>",
    ))
    fig_layout(fig, height=max(280, n*30))
    fig.update_xaxes(tickprefix="£", tickformat=",")
    return fig


def fig_dow(df):
    DOW = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow = (df.groupby("DayOfWeek")
             .agg(Orders=("InvoiceNo","nunique"), Revenue=("Revenue","sum"))
             .reindex(DOW).dropna().reset_index())
    bar_colors = [RED if row["Orders"] < dow["Orders"].median()*0.3
                  else BLUE for _, row in dow.iterrows()]
    fig = go.Figure(go.Bar(
        x=dow["DayOfWeek"].str[:3], y=dow["Orders"],
        marker=dict(color=bar_colors, line=dict(width=0)),
        hovertemplate="%{x}: %{y:,} orders<extra></extra>",
    ))
    fig_layout(fig, height=220)
    fig.update_yaxes(tickformat=",")
    return fig


def fig_category(df):
    cat = df.groupby("Category")["Revenue"].sum().reset_index()
    fig = go.Figure(go.Pie(
        labels=cat["Category"], values=cat["Revenue"],
        hole=0.6,
        marker=dict(colors=PALETTE, line=dict(color=CARD, width=2)),
        hovertemplate="%{label}: £%{value:,.0f} (%{percent})<extra></extra>",
        textinfo="none",
    ))
    fig_layout(fig, height=220)
    fig.update_layout(
        legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=10)),
    )
    return fig


def fig_rfm(rfm):
    seg = (rfm.groupby("Segment")
              .agg(Count=("Monetary","count"), Revenue=("Monetary","sum"))
              .reset_index().sort_values("Revenue", ascending=False))
    seg_colors = {"Champions":BLUE,"Loyal Customers":PINK,
                  "Growth Potential":GREEN,"At Risk":PURPLE,"Lost":RED}
    fig = go.Figure(go.Bar(
        x=seg["Segment"], y=seg["Revenue"],
        marker=dict(color=[seg_colors.get(s,GREY) for s in seg["Segment"]],
                    line=dict(width=0)),
        hovertemplate="%{x}: £%{y:,.0f}<extra></extra>",
        text=seg["Count"].astype(str) + " customers",
        textposition="outside", textfont=dict(size=10),
    ))
    fig_layout(fig, height=240)
    fig.update_yaxes(tickprefix="£", tickformat=",")
    return fig


def fig_monthly_orders(df):
    mo = (df.groupby(["Year","Month"])["InvoiceNo"].nunique().reset_index()
            .sort_values(["Year","Month"]))
    fig = go.Figure()
    colors = {2010: PURPLE, 2011: RED}
    for year, grp in mo.groupby("Year"):
        labels = pd.to_datetime(grp[["Year","Month"]].assign(day=1)).dt.strftime("%b")
        fig.add_trace(go.Scatter(
            x=labels, y=grp["InvoiceNo"],
            mode="lines+markers", name=str(year),
            line=dict(color=colors.get(year, GREY), width=2.5),
            marker=dict(size=5),
            hovertemplate="%{y:,} orders<extra>" + str(year) + "</extra>",
        ))
    fig_layout(fig, height=220)
    fig.update_yaxes(tickformat=",")
    return fig

app = Dash(__name__, title="E-Commerce Dashboard")

# Load data once at startup
print("Loading and cleaning data …")
if not os.path.exists(CSV):
    raise FileNotFoundError(
        f"\n\n  '{CSV}' not found.\n"
        "  Download from https://www.kaggle.com/datasets/carrie1/ecommerce-data\n"
        "  and place it in the same folder as dashboard.py\n"
    )
df  = load_data(CSV)
rfm = build_rfm(df)

total_rev  = df["Revenue"].sum()
orders_n   = df["InvoiceNo"].nunique()
customers  = df["CustomerID"].dropna().nunique()
aov        = df.groupby("InvoiceNo")["Revenue"].sum().mean()
n_countries= df["Country"].nunique()

rev_yr = df.groupby("Year")["Revenue"].sum()
yrs    = sorted(rev_yr.index.tolist())
yoy    = None
if len(yrs) >= 2:
    yoy = (rev_yr[yrs[-1]] - rev_yr[yrs[-2]]) / rev_yr[yrs[-2]] * 100

print("Building dashboard …")


app.layout = html.Div(style={"background":BG,"minHeight":"100vh",
                              "fontFamily":"Inter, system-ui, sans-serif"}, children=[


    html.Div([
        html.Div([
            html.H1("E-Commerce Analytics",
                    style={"margin":"0","fontSize":"22px","fontWeight":"500","color":"#2C2C2A"}),
            html.P("UK Online Retail · Jan 2010 – Dec 2011 · Kaggle Dataset",
                   style={"margin":"4px 0 0","fontSize":"12px","color":MUTED}),
        ]),
        html.Div([
            html.P("Country filter", style={"margin":"0 0 4px","fontSize":"11px","color":MUTED}),
            dcc.Dropdown(
                id="country-filter",
                options=[{"label":"All Countries","value":"ALL"}] +
                        [{"label":c,"value":c}
                         for c in sorted(df["Country"].unique())],
                value="ALL",
                clearable=False,
                style={"width":"200px","fontSize":"12px"},
            ),
        ]),
    ], style={"display":"flex","justifyContent":"space-between","alignItems":"flex-end",
               "padding":"24px 32px 16px","borderBottom":"0.5px solid #D3D1C7",
               "background":CARD}),


    html.Div(style={"padding":"24px 32px"}, children=[

        # KPI row
        html.Div(id="kpi-row", style={"display":"flex","gap":"12px","marginBottom":"20px","flexWrap":"wrap"}),

        # Revenue trend + Orders trend
        html.Div([
            section("Monthly Revenue Trend",    [dcc.Graph(id="fig-revenue",   config={"displayModeBar":False})],
                    {"flex":"3","marginRight":"16px","marginBottom":"0"}),
            section("Monthly Orders Trend",     [dcc.Graph(id="fig-orders",    config={"displayModeBar":False})],
                    {"flex":"2","marginBottom":"0"}),
        ], style={"display":"flex","marginBottom":"16px"}),

        # Products + Country
        html.Div([
            section("Top 10 Products by Revenue",  [dcc.Graph(id="fig-products", config={"displayModeBar":False})],
                    {"flex":"1","marginRight":"16px","marginBottom":"0"}),
            section("Revenue by Country",           [dcc.Graph(id="fig-country",  config={"displayModeBar":False})],
                    {"flex":"1","marginBottom":"0"}),
        ], style={"display":"flex","marginBottom":"16px"}),

        # DOW + Category + RFM
        html.Div([
            section("Sales by Day of Week",   [dcc.Graph(id="fig-dow",      config={"displayModeBar":False})],
                    {"flex":"1","marginRight":"16px","marginBottom":"0"}),
            section("Revenue by Category",   [dcc.Graph(id="fig-category", config={"displayModeBar":False})],
                    {"flex":"1","marginRight":"16px","marginBottom":"0"}),
            section("RFM Customer Segments", [dcc.Graph(id="fig-rfm",      config={"displayModeBar":False})],
                    {"flex":"2","marginBottom":"0"}),
        ], style={"display":"flex","marginBottom":"16px"}),

        # Insights
        section("Key Insights & Recommendations", [
            html.Div([
                html.Div([
                    html.Span("Critical", style={"background":L_RED,"color":RED,"fontSize":"10px",
                                                  "padding":"2px 8px","borderRadius":"4px","marginRight":"8px","fontWeight":"500"}),
                    html.Span("December 2010 revenue is low — likely data truncation. Verify completeness before forecasting.",
                              style={"fontSize":"13px","color":MUTED}),
                ], style={"padding":"10px 0","borderBottom":"0.5px solid #E8E6DF"}),
                html.Div([
                    html.Span("B2B Pattern", style={"background":L_BLUE,"color":BLUE,"fontSize":"10px",
                                                     "padding":"2px 8px","borderRadius":"4px","marginRight":"8px","fontWeight":"500"}),
                    html.Span("Saturday orders near zero, Sunday absent — this is a wholesale/B2B store with weekday-only buying behaviour.",
                              style={"fontSize":"13px","color":MUTED}),
                ], style={"padding":"10px 0","borderBottom":"0.5px solid #E8E6DF"}),
                html.Div([
                    html.Span("Opportunity", style={"background":PINK,"color":PINK,"fontSize":"10px",
                                                     "padding":"2px 8px","borderRadius":"4px","marginRight":"8px","fontWeight":"500"}),
                    html.Span("Netherlands, EIRE & Germany are top international markets and growing — targeted campaigns could double international revenue.",
                              style={"fontSize":"13px","color":MUTED}),
                ], style={"padding":"10px 0","borderBottom":"0.5px solid #E8E6DF"}),
                html.Div([
                    html.Span("Action", style={"background":PURPLE,"color":PURPLE,"fontSize":"10px",
                                                "padding":"2px 8px","borderRadius":"4px","marginRight":"8px","fontWeight":"500"}),
                    html.Span("November is the peak month both years. Stock up 8 weeks early and offer early-order discounts to wholesale clients.",
                              style={"fontSize":"13px","color":MUTED}),
                ], style={"padding":"10px 0"}),
            ])
        ]),

     
        html.P("Data source: Kaggle · carrie1/ecommerce-data  |  Built with Plotly Dash",
               style={"textAlign":"center","fontSize":"11px","color":MUTED,"marginTop":"8px"}),
    ]),
])


@app.callback(
    Output("kpi-row",      "children"),
    Output("fig-revenue",  "figure"),
    Output("fig-orders",   "figure"),
    Output("fig-products", "figure"),
    Output("fig-country",  "figure"),
    Output("fig-dow",      "figure"),
    Output("fig-category", "figure"),
    Output("fig-rfm",      "figure"),
    Input("country-filter","value"),
)
def update_all(country):
    d = df if country == "ALL" else df[df["Country"] == country]
    r = rfm  # RFM always global

    rev   = d["Revenue"].sum()
    ords  = d["InvoiceNo"].nunique()
    custs = d["CustomerID"].dropna().nunique()
    avg   = d.groupby("InvoiceNo")["Revenue"].sum().mean() if ords else 0

    kpis = [
        kpi_card("Total Revenue",    f"£{rev:,.0f}"),
        kpi_card("Total Orders",     f"{ords:,}"),
        kpi_card("Unique Customers", f"{custs:,}"),
        kpi_card("Avg Order Value",  f"£{avg:,.2f}"),
        kpi_card("Countries",        f"{d['Country'].nunique():,}"),
    ]
    if yoy is not None and country == "ALL":
        kpis.append(kpi_card("YoY Revenue", f"{yoy:+.1f}%", delta=None,
                              bg=L_PINK if yoy > 0 else L_RED))

    return (
        kpis,
        fig_revenue_trend(d),
        fig_monthly_orders(d),
        fig_top_products(d),
        fig_country_revenue(d),
        fig_dow(d),
        fig_category(d),
        fig_rfm(r),
    )

if __name__ == "__main__":
    try:
        app.run(debug=True)
    except SystemExit as e:
        print("Exited with code:", e.code)
