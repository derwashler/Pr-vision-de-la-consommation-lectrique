"""
=========================================================
  Projet 5 — Prévision de la Consommation Électrique
  Application de déploiement — Streamlit
  Étudiant : Mohamed Mahmoud El Atigh (C34621)
=========================================================

Démarrage :
    pip install streamlit joblib xgboost scikit-learn pandas numpy openpyxl
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import io

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Prévision Consommation Électrique",
    page_icon="⚡",
    layout="wide"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #f8f9fa; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none;
        padding: 0.6rem 2rem; border-radius: 8px;
        font-weight: bold; font-size: 16px;
        width: 100%; cursor: pointer;
    }
    .stButton > button:hover { opacity: 0.9; }
    .result-box {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 12px; padding: 20px;
        text-align: center; margin: 10px 0;
    }
    .result-box h2 { color: white; margin: 0; font-size: 2.5rem; }
    .result-box p  { color: white; margin: 0; font-size: 1rem; }
    .metric-card {
        background: white; border-radius: 8px;
        padding: 15px; border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,.08);
    }
</style>
""", unsafe_allow_html=True)


# ── Chargement des modèles ────────────────────────────────────
@st.cache_resource
def load_models():
    """Charge les modèles sauvegardés (une seule fois)."""
    try:
        models = {
            'Random Forest':        joblib.load('model_random_forest.pkl'),
            'XGBoost':              joblib.load('model_xgboost.pkl'),
            'Régression Linéaire':  joblib.load('model_linear_regression.pkl'),
        }
        scaler = joblib.load('scaler.pkl')
        with open('features.json') as f:
            features = json.load(f)
        return models, scaler, features, True
    except Exception as e:
        return None, None, None, False


models, scaler, features, loaded = load_models()


# ── Header ───────────────────────────────────────────────────
st.markdown("## ⚡ Prévision de la Consommation Électrique")
st.markdown(
    "**Projet 5 — M1 SSD 2025/2026 | Mohamed Mahmoud El Atigh (C34621)**"
)
st.markdown("---")

if not loaded:
    st.error(
        "⚠️ Modèles introuvables. Lancez d'abord le notebook pour entraîner "
        "et sauvegarder les modèles (model_random_forest.pkl, etc.)."
    )
    st.stop()

# Sélection du modèle
col_model, col_info = st.columns([1, 2])
with col_model:
    selected_model = st.selectbox(
        "🤖 Choisir le modèle de prédiction",
        list(models.keys())
    )
with col_info:
    model_desc = {
        'Random Forest':       "Ensemble d'arbres de décision. Robuste aux outliers. RMSE ~0.08",
        'XGBoost':             "Gradient Boosting optimisé. Meilleure performance générale. RMSE ~0.06",
        'Régression Linéaire': "Modèle baseline linéaire. Rapide et interprétable. RMSE ~0.15",
    }
    st.info(f"ℹ️ {model_desc[selected_model]}")

st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "🔍 Test Individuel — Saisie manuelle",
    "📂 Test par Fichier — Import CSV/Excel"
])


# ════════════════════════════════════════════════════════════
# TAB 1 : Saisie manuelle
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🔍 Prédiction manuelle")
    st.markdown(
        "Entrez les valeurs mesurées pour obtenir une prédiction instantanée "
        "de la consommation journalière moyenne."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**🔌 Puissance & Tension**")
        global_reactive = st.number_input(
            "Puissance réactive (kVAR)", 0.0, 2.0, 0.15, 0.01,
            help="Global_reactive_power — Puissance non productive"
        )
        voltage = st.number_input(
            "Tension moyenne (V)", 220.0, 260.0, 241.0, 0.1,
            help="Voltage — Tension du réseau"
        )
        global_intensity = st.number_input(
            "Intensité (A)", 0.0, 20.0, 4.6, 0.1,
            help="Global_intensity — Intensité du courant"
        )

    with c2:
        st.markdown("**📊 Sous-compteurs (Wh)**")
        sub1 = st.number_input(
            "Sous-compteur 1 (cuisine)", 0.0, 100.0, 1.0, 0.5,
            help="Sub_metering_1 — Cuisine (lave-vaisselle, four)"
        )
        sub2 = st.number_input(
            "Sous-compteur 2 (buanderie)", 0.0, 100.0, 1.0, 0.5,
            help="Sub_metering_2 — Buanderie (machine à laver)"
        )
        sub3 = st.number_input(
            "Sous-compteur 3 (chauffe-eau/clim)", 0.0, 200.0, 17.0, 1.0,
            help="Sub_metering_3 — Chauffe-eau & climatisation"
        )

    with c3:
        st.markdown("**📅 Informations temporelles**")
        day_of_week = st.selectbox(
            "Jour de la semaine",
            options=list(range(7)),
            format_func=lambda x: ['Lundi','Mardi','Mercredi','Jeudi',
                                    'Vendredi','Samedi','Dimanche'][x]
        )
        month = st.selectbox(
            "Mois",
            options=list(range(1, 13)),
            format_func=lambda x: ['Jan','Fév','Mar','Avr','Mai','Jun',
                                    'Jul','Aoû','Sep','Oct','Nov','Déc'][x-1],
            index=0
        )
        lag1 = st.number_input(
            "Consommation hier (kW)", 0.0, 15.0, 1.2, 0.1,
            help="lag_1 — Global_active_power de la veille"
        )

    # Features dérivées (calculées automatiquement)
    quarter   = (month - 1) // 3 + 1
    year      = 2025
    is_weekend = 1 if day_of_week >= 5 else 0
    lag7       = lag1 * 0.97   # Approximation
    lag30      = lag1 * 0.95
    rolling7   = lag1 * 0.98
    rolling30  = lag1 * 0.96

    st.markdown("---")
    if st.button("⚡ Lancer la prédiction", key="predict_individual"):
        input_data = pd.DataFrame([{
            'Global_reactive_power': global_reactive,
            'Voltage':               voltage,
            'Global_intensity':      global_intensity,
            'Sub_metering_1':        sub1,
            'Sub_metering_2':        sub2,
            'Sub_metering_3':        sub3,
            'day_of_week':           day_of_week,
            'month':                 month,
            'quarter':               quarter,
            'year':                  year,
            'is_weekend':            is_weekend,
            'lag_1':                 lag1,
            'lag_7':                 lag7,
            'lag_30':                lag30,
            'rolling_7':             rolling7,
            'rolling_30':            rolling30,
        }])

        # Vérifier l'ordre des colonnes
        input_data = input_data[features]
        model = models[selected_model]
        prediction = model.predict(input_data)[0]
        prediction = max(0, prediction)

        # Affichage du résultat
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f'<div class="result-box">'
                f'<p>⚡ Consommation prédite</p>'
                f'<h2>{prediction:.3f} kW</h2>'
                f'<p>Global Active Power journalier moyen</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        with r2:
            daily_kwh = prediction * 24
            st.markdown(
                f'<div class="result-box" style="background:linear-gradient(135deg,#f093fb,#f5576c)">'
                f'<p>🔋 Énergie estimée/jour</p>'
                f'<h2>{daily_kwh:.1f} kWh</h2>'
                f'<p>= {prediction:.3f} kW × 24h</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        with r3:
            monthly = daily_kwh * 30
            st.markdown(
                f'<div class="result-box" style="background:linear-gradient(135deg,#4facfe,#00f2fe)">'
                f'<p>📅 Estimation mensuelle</p>'
                f'<h2>{monthly:.0f} kWh</h2>'
                f'<p>≈ {monthly * 0.15:.1f} € (à 0,15 €/kWh)</p>'
                f'</div>',
                unsafe_allow_html=True
            )


# ════════════════════════════════════════════════════════════
# TAB 2 : Import fichier CSV/Excel
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📂 Prédictions par lot — Import CSV/Excel")
    st.markdown(
        "Importez un fichier avec les colonnes de features pour obtenir "
        "des prédictions multiples en une seule fois."
    )

    # Template à télécharger
    template_data = {feat: [0.0] * 3 for feat in features}
    template_df = pd.DataFrame(template_data)
    template_df['Global_reactive_power'] = [0.12, 0.15, 0.20]
    template_df['Voltage']               = [241.0, 238.5, 243.2]
    template_df['Global_intensity']      = [4.2, 5.1, 3.8]
    template_df['Sub_metering_1']        = [1.0, 0.0, 2.0]
    template_df['Sub_metering_2']        = [1.0, 2.0, 0.0]
    template_df['Sub_metering_3']        = [17.0, 18.0, 15.0]
    template_df['day_of_week']           = [0, 3, 6]
    template_df['month']                 = [1, 6, 12]
    template_df['quarter']               = [1, 2, 4]
    template_df['year']                  = [2025, 2025, 2025]
    template_df['is_weekend']            = [0, 0, 1]
    template_df['lag_1']                 = [1.2, 1.5, 0.9]
    template_df['lag_7']                 = [1.18, 1.48, 0.88]
    template_df['lag_30']                = [1.15, 1.45, 0.85]
    template_df['rolling_7']             = [1.19, 1.49, 0.89]
    template_df['rolling_30']            = [1.16, 1.46, 0.86]

    # Télécharger le template
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        "⬇️ Télécharger le template CSV",
        data=csv_template.encode('utf-8'),
        file_name="template_prediction.csv",
        mime="text/csv"
    )

    st.markdown("---")
    uploaded = st.file_uploader(
        "📁 Uploader votre fichier (CSV ou Excel)",
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded)
            else:
                df_upload = pd.read_excel(uploaded)

            st.success(f"✅ Fichier chargé : {len(df_upload)} lignes, {df_upload.shape[1]} colonnes")
            st.dataframe(df_upload.head(), use_container_width=True)

            # Vérification des colonnes
            missing_cols = [f for f in features if f not in df_upload.columns]
            if missing_cols:
                st.error(f"❌ Colonnes manquantes : {missing_cols}")
            else:
                if st.button("⚡ Prédire pour tout le fichier", key="predict_file"):
                    X_upload = df_upload[features].fillna(df_upload[features].median())
                    model = models[selected_model]
                    preds = model.predict(X_upload)
                    preds = np.maximum(0, preds)

                    df_upload['Prediction_GAP_kW']   = preds.round(4)
                    df_upload['Energie_journaliere_kWh'] = (preds * 24).round(2)
                    df_upload['Estimation_mensuelle_kWh'] = (preds * 24 * 30).round(1)

                    st.success(f"✅ {len(preds)} prédictions effectuées !")
                    st.dataframe(df_upload[['Prediction_GAP_kW',
                                            'Energie_journaliere_kWh',
                                            'Estimation_mensuelle_kWh']].describe(),
                                 use_container_width=True)
                    st.dataframe(df_upload, use_container_width=True)

                    # Export CSV
                    csv_out = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Télécharger les résultats (CSV)",
                        data=csv_out,
                        file_name="predictions_resultats.csv",
                        mime="text/csv"
                    )

                    # Export Excel
                    excel_buf = io.BytesIO()
                    with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                        df_upload.to_excel(writer, index=False, sheet_name='Prédictions')
                    st.download_button(
                        "⬇️ Télécharger les résultats (Excel)",
                        data=excel_buf.getvalue(),
                        file_name="predictions_resultats.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        except Exception as e:
            st.error(f"❌ Erreur : {e}")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Projet 5 | M1 SSD 2025/2026 | FST** — "
    "Étudiant : Mohamed Mahmoud El Atigh (C34621) | "
    "Prof. Ezyn SEGNANE"
)

try:
    import tensorflow as tf
    from tensorflow import keras
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False