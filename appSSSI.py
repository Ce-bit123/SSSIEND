import streamlit as st
import shap
from streamlit_shap import st_shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Title
st.header("Risk prediction of early neurological deterioration within 72 hours  in single small subcortical infarct")

# Input bar 1
# Input bar 2
NIHSS score = st.number_input("Enter NIHSS score on admission")
#NIHSS_score_after_thrombolysis1 = st.number_input("Enter NIHSS score after thrombolysis")
hemoglobin = st.number_input("Enter Hemoglobin(g/L)")
age = st.number_input("Enter Age(years)")
#fasting_blood_glucose= st.number_input("Enter Fasting blood glucose")
# Dropdown input
posterior type = st.selectbox("Whether it conforms to the posterior type", ("Yes", "No"))
onset to admission = st.number_input("Enter Time from onset to admission(hours)")

# If button is pressed
if st.button("Submit"):
    # Unpickle classifier
    clf = joblib.load("clfSSSI.pkl")

    # Store inputs into dataframe
    X = pd.DataFrame([[onset to admission, NIHSS score, hemoglobin,age,posterior type]],
                     columns=["onset to admission", "NIHSS score", "hemoglobin","age",
                       "posterior type"])
    X = X.replace(["Yes", "No"], [1, 0])

    # Get prediction
    prediction = clf.predict(X)[0]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    # f = plt.figure()
    # shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
    # f.savefig("shap_force_plot.png", bbox_inches='tight', dpi=600)
    # Output prediction
    # P = mpimg.imread("shap_force_plot.png")
    # st.image(P, caption="shap_force_plot", channels="RGB")
    # st_shap(shap.plots.waterfall(shap_values[0]), height=300)
    # st_shap(shap.plots.beeswarm(shap_values), height=300)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :]), height=200, width=700)
    if prediction == 0:
        st.text(f"This patient has a higher probability of Non-END within 72 hours")
    else:
        st.text(f"This patient has a higher probability of END within 72 hours")

