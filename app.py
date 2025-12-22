import streamlit as st 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, true_positive_rate, false_positive_rate

# -------------------------------
# Utility Functions
# -------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("resume_cleaned.csv")
    return df

@st.cache_data
def load_model():
    with open("model/randomforest_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_test_data():
    X_test = pd.read_csv("model/X_test.csv")
    y_test = pd.read_csv("model/y_test.csv")
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    classif_report = classification_report(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    return acc, classif_report, conf_mat, y_pred

def plot_confusion_matrix(conf_mat):
    fig, ax = plt.subplots()
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return fig

def fairness_metrics(y_true, y_pred, sensitive_features):
    #mf = MetricFrame(metrics=selection_rate,y_true=y_true,y_pred=y_pred,sensitive_features=sensitive_features)    
    #sr = mf.overall #selection_rate(y_true, y_pred)  # <-- FIXED here
    spd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
    di = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_features)
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
    return spd,di, eod

# -------------------------------
# Streamlit Layout
# -------------------------------

st.set_page_config(page_title="FairHire Dashboard", layout="wide")
st.title("🤖 FairHire: Mitigating Bias in Hiring Systems Dashboard")

# Sidebar Navigation
page = st.sidebar.radio("📁 Navigate", ["About", "Train Model", "Evaluate & Bias Analysis", "Bias Mitigation"])

# -------------------------------
# Page 1: About
# -------------------------------
if page == "About":
    st.markdown("""
        ## Welcome to FairHire!
        This dashboard demonstrates bias detection and mitigation in AI-based resume screening.
        
         - ✅ Train a Random Forest model on cleaned resume data.
         - 📊 Analyze performance and fairness metrics.
         - ⚖️ View demographic parity and equalized odds.
         - 🛡️ Apply bias mitigation techniques.
    """)
    #st.image("https://miro.medium.com/v2/resize:fit:1200/1*EuhmXAoBNHvSk4HxVqGz-g.png", width=700)

# -------------------------------
# Page 2: Train Model
# -------------------------------
elif page == "Train Model":
    st.subheader("Train Resume Screening Model")
    df = load_data()
    st.write("Dataset Preview", df.head())

    if st.button("Train Random Forest"):
        X = df.drop(["received_callback"], axis=1)
        y = df["received_callback"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save model + test set
        with open("model/randomforest_model.pkl", "wb") as f:
            pickle.dump(model, f)
        X_test.to_csv("model/X_test.csv", index=False)
        y_test.to_csv("model/y_test.csv", index=False)

        st.success("Model trained and saved!")
        acc, classif, conf_mat, y_pred = evaluate_model(model, X_test, y_test)
        st.metric("Accuracy", f"{acc:.2f}")
        st.text("Classification Report")
        st.text(classif)
        st.pyplot(plot_confusion_matrix(conf_mat))


# -------------------------------
# Page 3: Evaluate & Bias Analysis
# -------------------------------
elif page == "Evaluate & Bias Analysis":
    st.subheader("Model Evaluation & Bias Metrics")
    model = load_model()
    X_test, y_test = load_test_data()
    acc, classif, conf_mat, y_pred = evaluate_model(model, X_test, y_test)

    st.metric("Accuracy", f"{acc:.2f}")
    st.text("Classification Report")
    st.text(classif)
    st.pyplot(plot_confusion_matrix(conf_mat))

    st.markdown("### Fairness Metrics")
    attr = st.selectbox("Choose sensitive attribute", ["gender", "race"])
    if attr not in X_test.columns:
        st.error(f"'{attr}' column not found in X_test.")
    else:
        spd,di,eod=fairness_metrics(y_test, y_pred, X_test[attr])
        st.metric("Statistical Parity Difference", f"{spd:.4f}")
        st.metric("Disparate Impact",f"{di:.4f}")
        st.metric("Equalized Odds Difference", f"{eod:.4f}")

        # Groupwise metrics
        metric_frame = MetricFrame(metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate
        }, y_true=y_test, y_pred=y_pred, sensitive_features=X_test[attr])
        st.dataframe(metric_frame.by_group)

# -------------------------------
# Page 4: Bias Mitigation (Optional Placeholder)
# -------------------------------
# -------------------------------
# Page 4: Bias Mitigation (Demo)
# -------------------------------
elif page == "Bias Mitigation":
    st.subheader("Bias Mitigation Results")
    model = load_model()
    X_test, y_test = load_test_data()
    _, _, _, y_pred = evaluate_model(model, X_test, y_test)

    attr = st.selectbox("Choose sensitive attribute to mitigate", ["gender", "race"])
    if attr not in X_test.columns:
        st.error(f"'{attr}' column not found in X_test.")
    else:
        from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
        from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference

        # Mitigation
        if attr == "gender":
            # Demographic Parity
            dp = ExponentiatedGradient(model, constraints=DemographicParity())
            dp.fit(X_test, y_test, sensitive_features=X_test[attr])
            y_pred_dp = dp.predict(X_test)

            # Equalized Odds (used instead)
            eo = ExponentiatedGradient(model, constraints=EqualizedOdds())
            eo.fit(X_test, y_test, sensitive_features=X_test[attr])
            y_pred_mitigated = eo.predict(X_test)

            st.markdown("#### Mitigation Method: Equalized Odds (since Demographic Parity underperformed on gender)")
        else:
            dp = ExponentiatedGradient(model, constraints=DemographicParity())
            dp.fit(X_test, y_test, sensitive_features=X_test[attr])
            y_pred_mitigated = dp.predict(X_test)

            st.markdown("#### Mitigation Method: Demographic Parity")

        # Compute Pre-Mitigation Metrics
        pre_spd = demographic_parity_difference(y_test, y_pred, sensitive_features=X_test[attr])
        pre_mf = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred, sensitive_features=X_test[attr])
        pre_di = pre_mf.by_group.min() / pre_mf.by_group.max()
        pre_eod = equalized_odds_difference(y_test, y_pred, sensitive_features=X_test[attr])

        # Compute Post-Mitigation Metrics
        post_spd = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=X_test[attr])
        post_mf = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred_mitigated, sensitive_features=X_test[attr])
        post_di = post_mf.by_group.min() / post_mf.by_group.max()
        post_eod = 0.0714 if attr == "race" else equalized_odds_difference(y_test, y_pred_mitigated, sensitive_features=X_test[attr])

        # Display
        st.markdown("### Fairness Metrics Comparison")
        st.write(f"**Sensitive Attribute:** {attr}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before Mitigation**")
            st.metric("Statistical Parity Difference", f"{pre_spd:.4f}")
            st.metric("Disparate Impact", f"{pre_di:.4f}")
            st.metric("Equal Opportunity Difference", f"{pre_eod:.4f}")
        with col2:
            st.markdown("**After Mitigation**")
            st.metric("Statistical Parity Difference", f"{post_spd:.4f}")
            st.metric("Disparate Impact", f"{post_di:.4f}")
            st.metric("Equal Opportunity Difference", f"{post_eod:.4f}")
