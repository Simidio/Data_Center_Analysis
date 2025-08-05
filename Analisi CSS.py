import streamlit as st     # Libreria per l'interfaccia web interattiva
import pandas as pd        # Libreria per manipolazione dati
import numpy as np         # Libreria per calcoli numerici
import plotly.express as px  # Libreria per grafici interattivi semplici
import plotly.graph_objects as go  # Libreria per grafici più personalizzabili
import os

# =====================================================================================================
# === STEP 1 - LOAD & FILTER DATA =====================================================================
# =====================================================================================================

# Percorso del file Excel e parametri base

# Costruisce il percorso assoluto in base alla posizione del file Python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "Hourly CSS - Actual - Analysis.xlsx")

SHEET_NAME = "Hourly CSS - Actual"
DATE_COL = "DateTime"
CSS_COL = "North CSS"
MAX_FEE_SAVING = 60  # €/MWh di saving massimo per il Data Center

# Titolo dell'app
st.title("📊 Risk & Margin Analysis for 'Behind-the-Meter' Data Center Supply (North CSS)")

#  Spiegazione per il lettore
st.markdown("""
This report analyses **historical Clean Spark Spread (CSS)** to determine the optimal price and volume 
combinations for supplying power directly to a data center (behind-the-meter), avoiding grid fees.  
We simulate **risk-adjusted prices**, **total margin heatmaps**, and **operational insights** to guide contract design.
""")

# Caricamento file Excel
try:
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
except FileNotFoundError:
    st.error(f"❌ File not found: {EXCEL_PATH}")
    st.stop()

# Conversione colonna data e pulizia dati
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df[[DATE_COL, CSS_COL]].dropna()

# Aggiungo colonna "Year" per filtro anni
df["Year"] = df[DATE_COL].dt.year
available_years = sorted(df["Year"].unique())[::-1]
year_options = ["All years"] + [str(y) for y in available_years]

# Filtro anno lato sidebar
selected_year = st.sidebar.selectbox("📅 Select Year", year_options)
if selected_year != "All years":
    df = df[df["Year"] == int(selected_year)]

# Periodo analizzato (ultimi 3 anni)
last_ts = df[DATE_COL].max()
start_period = last_ts - pd.DateOffset(years=3)
df = df[df[DATE_COL] >= start_period].copy()

st.markdown(f"**Period analyzed:** {start_period.date()} – {last_ts.date()}  \n**Total hours:** {len(df)}")


# =====================================================================================================
# === STEP 2 - SIDEBAR PARAMETERS =====================================================================
# =====================================================================================================

st.sidebar.header("⚙️ Parameters")
var_level = st.sidebar.selectbox("Confidence Level (VaR)", [0.95, 0.975, 0.99], index=0)
risk_metric = st.sidebar.radio("Risk Measure Type", ["VaR", "CVaR"], index=0)
css_safety_premium = st.sidebar.slider("CSS Safety Margin (€)", 0, 20, 5)
msl = st.sidebar.number_input("Min Stable Load (MW)", min_value=0, value=197)
max_capacity = st.sidebar.number_input("Max Capacity (MW)", min_value=msl, value=385)


# =====================================================================================================
# === STEP 3 - RISK & PRICE CALCULATION ===============================================================
# =====================================================================================================
st.markdown("---")

# Calcolo VaR / CVaR
css_values = df[CSS_COL]
var_threshold = css_values.quantile(1 - var_level)

if risk_metric == "VaR":
    var_abs = abs(var_threshold)
    risk_value = var_threshold
else:
    var_abs = abs(css_values[css_values <= var_threshold].mean())
    risk_value = css_values[css_values <= var_threshold].mean()

# Prezzo target dinamico (VaR + margine sicurezza)
css_target_price = var_abs + css_safety_premium

# Sconto concesso al DC
premium_dc = MAX_FEE_SAVING - css_target_price

# Output riepilogo
st.markdown("### 🎯 Suggested Contract Price")
st.markdown("""
The suggested price is calculated as the **positive VaR** (or CVaR) plus a user-defined safety margin.  
This ensures that the price covers the downside risk at the selected confidence level.
""")

# CSS DISTRIBUTION CHART
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=css_values, nbinsx=100,
    name="CSS Distribution",
    marker_color='lightskyblue', opacity=0.75
))

# Linea VaR
fig_hist.add_vline(x=risk_value, line_dash="dash", line_color="black",
                   annotation_text=f"{risk_metric} = {risk_value:.2f}", annotation_position="top left")

# Linea prezzo target
fig_hist.add_vline(x=css_target_price, line_dash="dot", line_color="green",
                   annotation_text=f"Target Price = {css_target_price:.2f}", annotation_position="top right")

# Soglie visive
for lvl in [-60, -40, -20, 0, 20, 40, 60, 80]:
    fig_hist.add_vline(x=lvl, line_dash="dot", line_color="gray",
                       annotation_text=f"{lvl} €", annotation_position="bottom right")

fig_hist.update_layout(
    height=500, margin=dict(l=40, r=40, t=40, b=40), bargap=0.05,
    title_text="📊 CSS Historical Distribution with Risk Markers"
)

st.plotly_chart(fig_hist, use_container_width=True)

st.metric(f"{risk_metric} at {int(var_level*100)}%", f"{risk_value:.2f} €/MWh")
st.markdown(f"**Suggested Price (Abs {risk_metric} + margin):** `{css_target_price:.2f} €/MWh`")
st.markdown(f"**Discount to DC vs. 60€ ref.:** `{premium_dc:.2f} €/MWh`")

# =====================================================================================================
# === STEP 4 - CSS DISTRIBUTION CHART =================================================================
# =====================================================================================================
st.markdown("---")

st.markdown("### 📈 Forward Year+1 CSS Analysis")
st.markdown("""
This section analyzes the **Forward CSS (EP)** for the next year (sheet 'FWD'), 
allowing a direct comparison with historical data.  
It helps understand whether current market expectations (forward prices) 
are aligned or diverge from historical risk thresholds (VaR, CVaR).
""")

try:
    df_fwd = pd.read_excel(EXCEL_PATH, sheet_name="FWD")  # Caricamento dati forward
except Exception as e:
    st.error(f"❌ Unable to load 'FWD' sheet: {e}")
    df_fwd = None

if df_fwd is not None and not df_fwd.empty:
    # Conversione data
    df_fwd["Data"] = pd.to_datetime(df_fwd["Data"])

    # Serie valori FWD
    fwd_values = df_fwd["CSS EP FWD"].dropna()

    # Calcolo VaR e CVaR sui forward
    fwd_var = fwd_values.quantile(1 - var_level)
    fwd_cvar = fwd_values[fwd_values <= fwd_var].mean()

    # Calcolo VaR e CVaR storico (per confronto)
    hist_var = css_values.quantile(1 - var_level)
    hist_cvar = css_values[css_values <= hist_var].mean()

    # === Istogramma distribuzione FWD
    fig_fwd = go.Figure()
    fig_fwd.add_trace(go.Histogram(
        x=fwd_values,
        nbinsx=50,
        name="FWD CSS EP",
        marker_color='lightskyblue',
        opacity=0.75
    ))

    # Linee VaR e CVaR FWD
    fig_fwd.add_vline(x=fwd_var, line_dash="dash", line_color="black",
                      annotation_text=f"FWD VaR = {fwd_var:.2f}", annotation_position="top left")
    fig_fwd.add_vline(x=fwd_cvar, line_dash="dot", line_color="red",
                      annotation_text=f"FWD CVaR = {fwd_cvar:.2f}", annotation_position="top right")

    fig_fwd.update_layout(
        title="📊 FWD CSS Distribution with VaR & CVaR",
        xaxis_title="CSS EP FWD (€/MWh)",
        yaxis_title="Frequency",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_fwd, use_container_width=True)

    # === Tabella comparativa storico vs forward
    comparison_df = pd.DataFrame({
        "Metric": ["VaR", "CVaR"],
        "Historical CSS": [round(hist_var, 2), round(hist_cvar, 2)],
        "Forward CSS": [round(fwd_var, 2), round(fwd_cvar, 2)]
    })

    st.markdown("### 📋 VaR & CVaR Comparison – Historical vs Forward")
    st.dataframe(comparison_df, use_container_width=True)

else:
    st.warning("⚠️ 'FWD' sheet is empty or not available in the Excel file.")


# =====================================================================================================
# === User-defined Price for Subsequent Analysis ======================================================
# =====================================================================================================
st.markdown("---")

st.markdown("### Set Your Target Price for Subsequent Analysis")
st.markdown("""
Select a **custom target price (€/MWh)** to override the default value based on VaR + margin.  
All subsequent charts and simulations will use this price as the reference threshold.
""")

# Slider per definire il nuovo prezzo di riferimento
css_target_price = st.slider(
    "Custom Target Price (€/MWh)",
    min_value=float(min(css_values.min(), fwd_values.min())),
    max_value=float(max(css_values.max(), fwd_values.max())),
    value=float(css_target_price),  # default = prezzo calcolato da VaR + margin
    step=0.5
)

# Aggiorna anche il premium DC in base al nuovo prezzo
premium_dc = MAX_FEE_SAVING - css_target_price

st.markdown(f"**📌 New Target Price:** `{css_target_price:.2f} €/MWh`")
st.markdown(f"**📌 Updated DC Premium:** `{premium_dc:.2f} €/MWh`")



# =====================================================================================================
# === STEP 5 - VOLUME vs PRICE vs DISCOUNT TABLE ======================================================
# =====================================================================================================
st.markdown("---")

price_by_volume = []
dc_demands = list(range(50, int(max_capacity) + 1, 25))

for dc_mw in dc_demands:
    residual_mw = max(msl - dc_mw, 0)
    prezzo_richiesto = css_target_price + (residual_mw / max_capacity) * abs(risk_value)
    sconto = MAX_FEE_SAVING - prezzo_richiesto
    price_by_volume.append({
        "DC Demand (MW)": dc_mw,
        "Suggested Price (€/MWh)": round(prezzo_richiesto, 2),
        "Discount to DC (€)": round(sconto, 2)
    })

df_price_volume = pd.DataFrame(price_by_volume)
st.markdown("### 📊 Volume vs Requested Price vs DC Discount")
st.markdown("""
This table shows how the **requested price** to the DC and the **applied discount**  
change with different contracted volumes.  
Higher volumes reduce market exposure and therefore allow for lower prices.
""")
st.dataframe(df_price_volume, use_container_width=True)


# =====================================================================================================
# === STEP 6 - TOTAL MARGIN HEATMAP (BREAK-EVEN) ======================================================
# =====================================================================================================
st.markdown("---")

css_range = np.linspace(-100, 100, 201)
results = []

for dc_mw in dc_demands:
    for css_market in css_range:
        residual_mw = max(msl - dc_mw, 0)
        margin_dc = dc_mw * (css_market + css_target_price)
        margin_residual = residual_mw * css_market
        total_margin = margin_dc + margin_residual
        results.append({
            'DC_MW': dc_mw,
            'CSS_market': css_market,
            'Total_Margin': total_margin
        })

df_margins = pd.DataFrame(results)
pivot = df_margins.pivot(index='CSS_market', columns='DC_MW', values='Total_Margin')

st.markdown("### 🔥 Total Margin Heatmap")
st.markdown("""
This heatmap shows the **total hourly margin** as a function of:  
- **DC contracted volume** (x-axis)  
- **Market CSS** (y-axis)  

The black markers indicate the **break-even points** where the total margin is close to zero.
""")
fig_margin = go.Figure()

fig_margin.add_trace(go.Heatmap(
    z=pivot.values,
    x=pivot.columns,
    y=pivot.index,
    colorscale='RdYlGn',
    colorbar=dict(title="Total Margin (€ / h)"),
    zmid=0,
    hovertemplate='DC Demand: %{x} MW<br>CSS Market: %{y} €/MWh<br>Margin: %{z:.0f} €/h<extra></extra>'
))

# Linea break-even
break_even_points = []
for dc_mw in dc_demands:
    subset = df_margins[df_margins['DC_MW'] == dc_mw]
    if not subset.empty:
        closest = subset.iloc[(subset['Total_Margin']).abs().argsort()[:1]]
        break_even_css = closest['CSS_market'].values[0]
        break_even_points.append((dc_mw, break_even_css))

fig_margin.add_trace(go.Scatter(
    x=[x for x, _ in break_even_points],
    y=[y for _, y in break_even_points],
    mode='lines+markers',
    line=dict(color='black', width=2),
    name="Break-even Line"
))

fig_margin.update_layout(
    xaxis_title="DC Demand (MW)",
    yaxis_title="Market CSS (€/MWh)",
    margin=dict(l=60, r=20, t=40, b=40),
    height=600
)

st.plotly_chart(fig_margin, use_container_width=True)


# =====================================================================================================
# === STEP 7 - KEY OFFER COMBINATIONS (Dynamic Market CSS) ============================================
# =====================================================================================================
st.markdown("---")

st.markdown("### 🎯 Set Your Target Market CSS")
# Slider per simulare un CSS di mercato di riferimento
css_market_sim = st.slider(
    label="Simulated Market CSS (€/MWh)",
    min_value=-50.0, max_value=150.0,
    value=float(css_target_price), step=1.0
)

st.markdown("### 📋 Key Offer Combinations (Based on Selected Market CSS)")
st.markdown("""
The suggested price already **includes the CSS Safety Margin** you set earlier in the parameters.  
""")

# Lista combinazioni prezzo–volume con margine calcolato
offer_combinations = []

for dc_mw in dc_demands:
    residual_mw = max(msl - dc_mw, 0)
    suggested_price = css_market_sim + css_safety_premium + (residual_mw / max_capacity) * abs(risk_value)
    margin_dc = dc_mw * (css_market_sim + css_target_price)
    margin_residual = residual_mw * css_market_sim
    total_margin = margin_dc + margin_residual
    offer_combinations.append({
        "DC Volume (MW)": dc_mw,
        "Suggested Price (€/MWh)": round(suggested_price, 2),
        "Expected Total Margin (€/h)": round(total_margin, 1),
        "Residual Volume (MW)": residual_mw
    })

df_offers = pd.DataFrame(offer_combinations)
st.dataframe(df_offers, use_container_width=True)


# =====================================================================================================
# === STEP 8 - HEATMAP: AVERAGE CSS BY HOUR AND MONTH ==================================================
# =====================================================================================================
st.markdown("---")

st.markdown("### 🔥 Heatmap: Average CSS by Hour of Day and Month")
st.markdown("""
This chart shows the **average CSS profile** by hour of the day and month of the year.  
It helps identify seasonal and intraday patterns that can influence contract design.
""")
st.markdown(f"The break-even line shows, for each level of DC demand, the minimum market CSS required to maintain zero margin.")
st.markdown(f"Total Margin = DC Volume × (Current CSS + DC Premium) + Residual Volume × Current CSS")

# Creo colonne ora e mese
df["hour"] = df[DATE_COL].dt.hour
df["month"] = df[DATE_COL].dt.month

# Creo pivot con media CSS per (ora, mese)
heatmap_monthly = df.pivot_table(index="hour", columns="month", values=CSS_COL, aggfunc="mean")

# Rinomino mesi con abbreviazione (Jan, Feb...)
month_labels = {i: pd.to_datetime(f"2024-{i:02d}-01").strftime('%b') for i in heatmap_monthly.columns}
heatmap_monthly.rename(columns=month_labels, inplace=True)

# Grafico heatmap
fig_monthly = px.imshow(
    heatmap_monthly,
    labels=dict(x="Month", y="Hour of Day", color="Avg CSS (€/MWh)"),
    color_continuous_scale="RdBu_r",
    aspect="auto"
)
fig_monthly.update_layout(
    yaxis=dict(dtick=1),
    xaxis=dict(tickmode="array", tickvals=list(month_labels.values()))
)
st.plotly_chart(fig_monthly, use_container_width=True)


# =====================================================================================================
# === STEP 9 - OPERATIONAL INSIGHTS: CRITICAL HOURS ====================================================
# =====================================================================================================
st.markdown("---")

st.markdown("### ⚠️ Operational Insights: Critical Hours")
st.markdown("""
This table identifies the **hours of the day** with higher operational risk:  
- Hours when CSS is below the selected **VaR threshold** more than X% of the time  
- Hours when total margin is negative more than Y% of the time
""")


# Soglia di attivazione impostata dall'utente (usiamo VaR come base)
activation_threshold = risk_value
st.markdown(f"** % hours with CSS < VaR (Filtered by threshold of VaR > 5%): {activation_threshold:.2f} €/MWh**")
st.markdown(f"** % hours with Margin < 0 (Filtered by threshold of negative hours > 5%)**")

# Ore con CSS < soglia
ore_critiche_css = df[df[CSS_COL] < activation_threshold].groupby("hour").size()
ore_critiche_css = (ore_critiche_css / df.groupby("hour").size()) * 100
ore_critiche_css = ore_critiche_css[ore_critiche_css > 5]  # filtro >5%

# Ore con margine negativo
df["margine_orario"] = df[CSS_COL] * max_capacity + premium_dc * max_capacity
ore_margine_neg = df[df["margine_orario"] < 0].groupby("hour").size()
ore_margine_neg = (ore_margine_neg / df.groupby("hour").size()) * 100
ore_margine_neg = ore_margine_neg[ore_margine_neg > 5]  # filtro >5%

# Tabella ore critiche
df_insight = pd.DataFrame({
    "Hour": list(range(24)),
    "% hours with CSS < Threshold": ore_critiche_css.reindex(range(24), fill_value=0).round(1),
    "% hours with Margin < 0": ore_margine_neg.reindex(range(24), fill_value=0).round(1)
})

st.dataframe(df_insight.style
    .background_gradient(axis=0, subset=["% hours with CSS < Threshold"], cmap="Oranges")
    .background_gradient(axis=0, subset=["% hours with Margin < 0"], cmap="Reds")
    .format("{:.1f}%")
)


# =====================================================================================================
# === STEP 10 - ACTIVABLE HOURS ANALYSIS ==============================================================
# =====================================================================================================
st.markdown("---")

st.markdown("### 🕐 Activable Hours per Hour of Day")
st.markdown("""
This chart shows the **percentage of activable hours** in each hour of the day,  
based on the selected risk threshold and margin conditions.
""")

# Ore attivabili = CSS > soglia e no ore con alto rischio di margine negativo
df["activable"] = (df[CSS_COL] > activation_threshold) & \
                  (~df["hour"].isin(df_insight[df_insight["% hours with Margin < 0"] > 35]["Hour"]))

activable_hours = df[df["activable"]]
ore_attivabili = len(activable_hours)
ore_anno = len(df)
perc_ore_attivabili = ore_attivabili / ore_anno * 100
volume_anno_mwh = ore_attivabili * max_capacity

# Ricavo medio EP e saving DC
ricavo_ep_medio = activable_hours[CSS_COL].mean() * max_capacity
saving_dc_medio = premium_dc * max_capacity

# Output riepilogo
st.markdown(f"""
**🔢 Summary:**  
- Activable hours: `{ore_attivabili} h` ({perc_ore_attivabili:.1f}%)  
- Deliverable volume: `{volume_anno_mwh:.0f} MWh`  
- Avg. EP revenue/hour: `{ricavo_ep_medio:.2f} €/h`  
- Avg. DC saving/hour: `{saving_dc_medio:.2f} €/h`
""")

# Grafico % ore attivabili per ora del giorno
attivabili_per_ora = activable_hours.groupby("hour").size() / df.groupby("hour").size() * 100
fig_attivabili = px.bar(
    x=attivabili_per_ora.index,
    y=attivabili_per_ora.values,
    labels={"x": "Hour", "y": "% Activable Hours"},
    title="% Activable Hours per Hour of Day"
)
st.plotly_chart(fig_attivabili, use_container_width=True)


# =====================================================================================================
# === STEP 12 - Ex-Post Margin Analysis with Selected Year Prices =====================================
# =====================================================================================================
st.markdown("---")

st.markdown("## 📊 Ex-Post Margin Analysis (Selected Year)")
st.markdown("""
This section allows you to **simulate the margins** that would have been generated by applying  
the selected **target price** to a chosen DC contracted volume,  
using **real hourly prices** from a specific year.
""")

#  Ricarico i dati completi senza filtri precedenti
df_all_years = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
df_all_years[DATE_COL] = pd.to_datetime(df_all_years[DATE_COL])
df_all_years = df_all_years[[DATE_COL, CSS_COL]].dropna()

#  Scelta anno per analisi ex-post
years_available = sorted(df_all_years[DATE_COL].dt.year.unique())
year_expost = st.selectbox("Select Year for Ex-Post Analysis", years_available, index=len(years_available)-1)

#  Filtro solo l'anno selezionato
df_current = df_all_years[df_all_years[DATE_COL].dt.year == year_expost].copy()

#  Scelta volume DC
dc_mw_expost = st.slider(
    "Select DC Volume for Ex-Post Simulation (MW)",  # testo principale
    min_value=50,
    max_value=int(max_capacity),
    value=int(max_capacity),
    step=25
)

# Testo esplicativo aggiuntivo
st.markdown(
    "_ℹ️ If DC volume ≥ Min Stable Load (MSL), then residual volumes = 0 and all capacity is sold to the DC at the contract price._"
)

#  Calcolo margini
residual_mw_expost = max(msl - dc_mw_expost, 0)  # MW residui esposti al mercato
df_current["DC_Margin"] = dc_mw_expost * css_target_price
df_current["Residual_Margin"] = residual_mw_expost * df_current[CSS_COL]
df_current["Total_Margin"] = df_current["DC_Margin"] + df_current["Residual_Margin"]

#  KPI aggregati
total_margin_year = df_current["Total_Margin"].sum()
avg_margin_hour = df_current["Total_Margin"].mean()

#  Output riepilogo
st.markdown(f"""
** Year analysed:** {year_expost}  
** DC Volume:** {dc_mw_expost} MW  
** Total Margin (Margin = (DC_MW × target price) + (Residual_MW × Market CSS)):** `{total_margin_year:,.0f} €`  
** Avg Hourly Margin:** `{avg_margin_hour:.2f} €/h`
""")

#  Grafico temporale margine orario
fig_expost = px.line(
    df_current,
    x=DATE_COL,
    y="Total_Margin",
    title=f"📈 Hourly Total Margin - Ex-Post Simulation ({year_expost})",
    labels={DATE_COL: "Date", "Total_Margin": "Margin (€/h)"}
)
st.plotly_chart(fig_expost, use_container_width=True)




# =====================================================================================================
# === STEP 13 - STREAMLIT RUN =========================================================================
# =====================================================================================================

# streamlit run "Data Center Analysis/Analisi CSS.py"