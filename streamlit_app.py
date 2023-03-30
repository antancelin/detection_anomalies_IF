import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

# Définir l'interface utilisateur
st.title("Détection d'anomalies avec l'Isolation Forest")
contamination = st.slider("Taux de contamination", min_value=0.0, max_value=0.5, value=0.02, step=0.01)
n_estimators = st.slider("Nombre d'estimateurs", min_value=1, max_value=100, value=100, step=1)
uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=["csv"])

# Charger les données CSV et détecter les anomalies
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination)
    model.fit(data)
    y_pred = model.predict(data)
    data['anomaly'] = y_pred
    st.write(data)