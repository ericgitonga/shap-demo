import streamlit as st
from streamlit_shap import st_shap
import streamlit_analytics

import pandas as pd
import shap
from sklearn.model_selection import train_test_split
import sklearn


st.title("Explainer Prototype")

X, y = shap.datasets.boston()
X_display, y_display = shap.datasets.boston(display=True)

model = sklearn.linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X, y)

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

with streamlit_analytics.track():
    index = st.selectbox("Select the row to display", list(range(1, len(X)+1)), index=0)
    st.markdown("### Row {}".format(index))
    
    st.markdown("## Beeswarm Plot")
    st_shap(shap.plots.beeswarm(shap_values[index-1:index]), height=300)
    
    st.markdown("## Waterfall Plot")
    st_shap(shap.plots.waterfall(shap_values[index]), height=300)
    
    st.markdown("## Force Plot")
    shap_values = explainer.shap_values(X)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[index,:], X_display.iloc[0,:]), height=200, width=1000)
