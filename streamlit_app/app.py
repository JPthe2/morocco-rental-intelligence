"""Morocco Rental Market Intelligence — Streamlit app.

Self-contained: reads the static data.json snapshot and runs the trained XGBoost
model in-process (no FastAPI services needed). Five tabs mirror the existing
HTML dashboard: Overview, Listings, Predict Price, Deals, Chat Assistant.

Run locally:  streamlit run streamlit_app/app.py
Deploy:       Streamlit Community Cloud, main file = streamlit_app/app.py
"""

import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import assistant  # noqa: E402
import market_engine  # noqa: E402
from data_loader import file_status, load_dashboard_data, load_metrics, load_model  # noqa: E402

# Streamlit secrets (e.g. OPENROUTER_API_KEY / GEMINI_API_KEY) -> env for assistant.py
try:
    for key in ("OPENROUTER_API_KEY", "OPENROUTER_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL", "LLM_PROVIDER"):
        if not os.getenv(key) and key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

st.set_page_config(
    page_title="Morocco Rental Market Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

TIER_COLORS = {
    "Value": "color: #199e70; font-weight: 600;",
    "Mid-range": "color: #2a78d6; font-weight: 600;",
    "Family-oriented": "color: #4a3aa7; font-weight: 600;",
    "Premium": "color: #c98500; font-weight: 600;",
    "Balanced": "color: #52514e; font-weight: 600;",
}


def fmt_mad(n) -> str:
    if n is None:
        return "—"
    return f"{round(n):,} MAD"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🏠 Morocco Rental Market Intelligence")
st.sidebar.caption(
    "Streamlit port of the existing dashboard — self-contained, no FastAPI services required."
)

with st.sidebar.expander("Backing files", expanded=False):
    for label, ok in file_status().items():
        st.write(f"{'✅' if ok else '❌'} {label}")

model_ok = False
try:
    load_model()
    model_ok = True
except Exception as exc:
    st.sidebar.error(f"Model not loadable: {exc}")

if model_ok:
    try:
        meta = market_engine.model_meta()
        with st.sidebar.expander("Model info", expanded=False):
            st.write(f"Model: **{meta['model_name']}**")
            st.write(f"Trained on: **{meta['n_train_samples']:,} listings**")
            st.write(f"Residual std: **{meta['residual_std']:,.0f} MAD**")
            if meta.get("metrics"):
                m = meta["metrics"].get("xgboost", {})
                st.write(f"Test R²: **{m.get('r2', '—')}**")
                st.write(f"Test RMSE: **{m.get('rmse', '—'):,.0f}** MAD")
    except Exception:
        pass

st.sidebar.markdown("---")
st.sidebar.caption("Backed by the live scraped dataset (agenz.ma + avito.ma).")

# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------

try:
    DASH = load_dashboard_data()
    SUMMARY = DASH["summary"]
    LISTINGS = pd.DataFrame(DASH["listings"])
    data_ok = True
except Exception as exc:
    SUMMARY = None
    LISTINGS = pd.DataFrame()
    data_ok = False
    st.error(f"Could not load frontend/data.json: {exc}")

tab_overview, tab_listings, tab_predict, tab_deals, tab_chat = st.tabs(
    ["Overview", "Listings", "Predict Price", "Deals", "Chat Assistant"]
)

# ---------------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------------

with tab_overview:
    if not data_ok:
        st.warning("Dashboard data is unavailable — check the sidebar 'Backing files'.")
    else:
        try:
            generated = datetime.fromisoformat(SUMMARY["generatedAt"].replace("Z", "+00:00"))
        except Exception:
            generated = None
        st.caption(
            f"Snapshot generated {generated.strftime('%Y-%m-%d %H:%M') if generated else SUMMARY['generatedAt']} · "
            f"{SUMMARY['totalListings']:,} total listings"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total listings", f"{SUMMARY['totalListings']:,}",
                  f"{SUMMARY['bySite'].get('agenz', 0):,} agenz · {SUMMARY['bySite'].get('avito', 0):,} avito")
        c2.metric("Median rent", fmt_mad(SUMMARY["medianRent"]),
                  f"avg {fmt_mad(SUMMARY['avgRent'])} (skewed)")
        c3.metric("Rent range", fmt_mad(SUMMARY["minRent"]),
                  f"up to {fmt_mad(SUMMARY['maxRent'])}")
        c4.metric("Cities covered", f"{len(SUMMARY['cityStats'])}",
                  "with priced listings")

        col_chart, col_hist = st.columns(2)
        with col_chart:
            st.subheader("Median rent by city (top 12)")
            top = pd.DataFrame(SUMMARY["cityStats"][:12])
            st.bar_chart(top.set_index("city")["medianRent"])
        with col_hist:
            st.subheader("Rent distribution")
            hist = pd.DataFrame(SUMMARY["histogram"])
            st.bar_chart(hist.set_index("range")["count"])

        col_bed, col_qual = st.columns(2)
        with col_bed:
            st.subheader("By bedroom count")
            st.dataframe(
                pd.DataFrame(SUMMARY["bedroomStats"]),
                column_config={"bedrooms": "Bedrooms", "count": "Listings", "avgRent": "Avg rent (MAD)"},
                hide_index=True,
                width="stretch",
            )
        with col_qual:
            st.subheader("Data quality")
            a = SUMMARY.get("amenityCounts", {})
            bysrc = SUMMARY.get("bySourceStats", {})
            quality = pd.DataFrame({
                "Item": [
                    "Listings with a confirmed price",
                    "Price from listing field",
                    "No price found (queued for ML)",
                    "Missing / other",
                    "Elevator / Parking / Terrace",
                    "Balcony / Garden (agenz only)",
                ],
                "Count": [
                    f"{SUMMARY['withPriceCount']:,} / {SUMMARY['totalListings']:,}",
                    f"{bysrc.get('listing', 0):,}",
                    f"{bysrc.get('no_price_found', 0):,}",
                    f"{bysrc.get('missing', 0):,}",
                    f"{a.get('has_elevator', 0):,} / {a.get('has_parking', 0):,} / {a.get('has_terrace', 0):,}",
                    f"{a.get('has_balcony', 0):,} / {a.get('has_garden', 0):,}",
                ],
            })
            st.dataframe(quality, hide_index=True, width="stretch")

        st.subheader("Neighborhood tiers (unsupervised clustering)")
        if model_ok:
            try:
                tiers = market_engine.neighborhood_tiers()
                if tiers.get("error"):
                    st.info(tiers["error"])
                else:
                    tiers_df = pd.DataFrame(tiers["neighborhoods"])
                    st.caption(
                        f"{len(tiers_df)} neighborhoods grouped into {tiers['n_clusters']} tiers "
                        "by price/m², bedroom count and amenity mix (KMeans)."
                    )
                    styled = tiers_df.style.map(
                        lambda v: TIER_COLORS.get(v, ""), subset=["tier"]
                    )
                    st.dataframe(
                        styled,
                        column_config={
                            "listing_count": "Listings",
                            "median_price_per_m2": "MAD/m²",
                            "avg_bedrooms": "Avg bed",
                            "avg_amenity_count": "Avg amen.",
                        },
                        hide_index=True,
                        width="stretch",
                    )
            except Exception as exc:
                st.warning(f"Could not compute neighborhood tiers: {exc}")
        else:
            st.warning("Neighborhood tiers need the trained model, which isn't loadable.")

# ---------------------------------------------------------------------------
# LISTINGS
# ---------------------------------------------------------------------------

with tab_listings:
    if not data_ok:
        st.warning("Dashboard data is unavailable.")
    else:
        st.subheader("Natural-language filter")
        nl_col1, nl_col2 = st.columns([3, 1])
        nl_query = nl_col1.text_input(
            "Describe what you want",
            placeholder="e.g. 3-bedroom furnished apartments in Rabat with parking under 7000 MAD",
            label_visibility="collapsed",
        )
        nl_apply = nl_col2.button("Apply NL filter", width="stretch")

        if nl_apply and nl_query.strip():
            f = assistant.parse_nl_filter(nl_query)
            if f["city"]:
                st.session_state["list_city"] = f["city"]
            if f["max_price"] is not None:
                st.session_state["list_max_price"] = float(f["max_price"])
            if f["min_bedrooms"] is not None:
                st.session_state["list_min_bed"] = str(f["min_bedrooms"])
            if not any(v is not None for v in (f["city"], f["max_price"], f["min_bedrooms"])):
                st.info("Couldn't extract a city, price or bedroom count from that — try the plain filters below.")

        st.subheader("Filters")
        cities = sorted({c for c in LISTINGS["city"].dropna()})
        fc1, fc2, fc3, fc4 = st.columns(4)
        city = fc1.selectbox("City", ["All"] + cities, key="list_city")
        max_price = fc2.number_input("Max rent (MAD)", min_value=0, value=0, step=500, key="list_max_price")
        min_bed = fc3.selectbox("Min bedrooms", ["Any", "1", "2", "3", "4+"], key="list_min_bed")
        keyword = fc4.text_input("Keyword (neighborhood/city)", key="list_keyword")

        rows = LISTINGS.copy()
        if city != "All":
            rows = rows[rows["city"] == city]
        if max_price and max_price > 0:
            rows = rows[rows["rent_price"] <= max_price]
        if min_bed != "Any":
            n = 4 if min_bed == "4+" else int(min_bed)
            rows = rows[rows["bedrooms"] >= n]
        if keyword:
            kw = keyword.lower()
            rows = rows[
                rows["city"].astype(str).str.lower().str.contains(kw, na=False)
                | rows["neighborhood"].astype(str).str.lower().str.contains(kw, na=False)
            ]
        rows = rows.head(300)

        st.caption(f"{len(rows)} shown (of {len(LISTINGS)})")
        st.dataframe(
            rows,
            column_config={
                "rent_price": st.column_config.NumberColumn("Rent (MAD)", format="%d"),
                "surface_m2": st.column_config.NumberColumn("m²", format="%d"),
                "price_per_m2": st.column_config.NumberColumn("MAD/m²", format="%.1f"),
                "url": st.column_config.LinkColumn("Link"),
            },
            hide_index=True,
            width="stretch",
        )

# ---------------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------------

with tab_predict:
    if not model_ok:
        st.warning("Prediction needs the trained model, which isn't loadable.")
    else:
        st.subheader("Predict monthly rent")

        try:
            known_cities = sorted(market_engine._known_cities())
        except Exception:
            known_cities = sorted({c for c in SUMMARY["cityStats"]}) if SUMMARY else []
        city_opts = known_cities + ["Other (type below)"]

        pc1, pc2 = st.columns(2)
        city_choice = pc1.selectbox("City", city_opts, index=0)
        city_text = pc1.text_input("City (free text)", placeholder="Only if you picked 'Other'") if city_choice.startswith("Other") else None
        city = city_text.strip() if city_text else city_choice
        neighborhood = pc1.text_input("Neighborhood (optional)")
        surface = pc2.number_input("Surface (m²)", min_value=0.0, value=90.0, step=10.0)
        bedrooms = pc2.slider("Bedrooms", 0, 6, 2)
        bathrooms = pc2.slider("Bathrooms", 0, 5, 1)
        furnished = pc2.selectbox("Furnished", ["Unknown", "Yes", "No"])
        amenities = pc2.multiselect(
            "Amenities (counts toward the model)",
            ["ascenseur", "parking", "terrasse", "balcon", "jardin"],
        )

        predict_clicked = st.button("Predict rent", type="primary", width="stretch")

        if predict_clicked:
            if not city:
                st.error("Please enter a city.")
            else:
                with st.spinner("Running model…"):
                    result = market_engine.predict_rent(
                        city=city,
                        neighborhood=neighborhood or None,
                        surface_m2=surface or None,
                        bedrooms=bedrooms or None,
                        bathrooms=bathrooms or None,
                        furnished=None if furnished == "Unknown" else furnished == "Yes",
                        amenities=amenities,
                    )
                low, high = result["confidence_interval_80pct"]
                col1, col2 = st.columns(2)
                col1.metric("Predicted rent", fmt_mad(result["predicted_rent_mad"]),
                            f"{result['model_used']} model")
                col2.metric("80% confidence range",
                            f"{fmt_mad(low)} – {fmt_mad(high)}")
                if result.get("warning"):
                    st.warning(result["warning"])
                if result.get("top_drivers"):
                    st.caption("Top price drivers (SHAP, impact on log price):")
                    for d in result["top_drivers"]:
                        st.write(f"- **{d['feature']}**: {d['impact_on_log_price']:+.4f}")

# ---------------------------------------------------------------------------
# DEALS
# ---------------------------------------------------------------------------

with tab_deals:
    if not model_ok:
        st.warning("Deals need the trained model, which isn't loadable.")
    else:
        st.subheader("Potential deals")
        st.caption(
            "Flags listings priced meaningfully below what the ML model predicts for a comparable "
            "property — genuinely outside its own 80% confidence band, not just below the point "
            "estimate. This model's R² is modest (~0.47), so treat this as a lead worth a second "
            "look, not a guarantee."
        )
        d1, d2 = st.columns(2)
        threshold = d1.slider("Discount threshold (%)", 5, 40, 15, 5)
        limit = d2.slider("Max deals shown", 10, 100, 50, 10)

        with st.spinner("Scoring listings…"):
            deals = market_engine.find_deals(threshold_pct=threshold, limit=limit)

        st.metric("Deals found", f"{deals['count']}")
        if deals["deals"]:
            df = pd.DataFrame(deals["deals"])
            df = df.rename(columns={
                "rent_price": "Actual (MAD)", "predicted_rent_mad": "Predicted (MAD)",
                "discount_pct": "Discount %", "surface_m2": "m²", "bedrooms": "Bed",
                "neighborhood": "Neighborhood", "city": "City", "url": "Link",
            })
            st.dataframe(
                df[["City", "Neighborhood", "Actual (MAD)", "Predicted (MAD)", "Discount %", "m²", "Bed", "Link"]],
                column_config={
                    "Actual (MAD)": st.column_config.NumberColumn(format="%d"),
                    "Predicted (MAD)": st.column_config.NumberColumn(format="%d"),
                    "Discount %": st.column_config.NumberColumn(format="%.1f"),
                    "Link": st.column_config.LinkColumn(),
                },
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No deals flagged at this threshold right now.")

# ---------------------------------------------------------------------------
# CHAT
# ---------------------------------------------------------------------------

with tab_chat:
    llm_on = assistant.llm_config() is not None
    st.subheader("Chat Assistant")
    st.caption(
        f"Mode: **{'LLM-backed (OpenRouter/Gemini)' if llm_on else 'local rule-based'}**. "
        "Ask about rents, neighborhoods, deals, or ask me to predict a price."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("e.g. What's the average rent in Casablanca?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-16:-1]
        ]
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = assistant.chat(prompt, history)
            st.markdown(reply["text"])
            if reply["degraded"]:
                st.caption("(degraded mode)")
        st.session_state.messages.append({"role": "assistant", "content": reply["text"]})
