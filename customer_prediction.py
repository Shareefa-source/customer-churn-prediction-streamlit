import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title='Customer Churn Prediction', layout='wide')

st.markdown("""
# 📊 Customer Churn Prediction App
*End-to-end churn analysis & prediction using a single ML pipeline*
""")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title('Navigation')
page = st.sidebar.radio('Select Page', ['Home', 'EDA', 'Visualisation', 'Modeling', 'Prediction'])

uploaded_file = st.sidebar.file_uploader(
    "Upload dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.warning('Please upload a dataset to continue')
    st.stop()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
if uploaded_file.name.endswith('csv'):
    df = pd.read_csv(uploaded_file)
elif uploaded_file.name.endswith('xlsx'):
    df = pd.read_excel(uploaded_file)

# --------------------------------------------------
# HOME
# --------------------------------------------------
if page == 'Home':
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        ### 🎯 Objective
        Predict customer churn and support proactive retention strategies.
        """)

    with col2:
        st.success("""
        ### 📈 Features
        - Data exploration
        - Visualization
        - ML model training
        """)

    with col3:
        st.warning("""
        ### 🔮 Prediction
        Predict churn probability for new customers.
        """)

    st.markdown("""
    ### 🧭 How to Use
    1. Upload dataset
    2. Explore data
    3. Train model
    4. Predict churn
    """)

# --------------------------------------------------
# EDA
# --------------------------------------------------
elif page == 'EDA':
    st.header('📊 Exploratory Data Analysis')

    st.subheader('Dataset Preview')
    st.dataframe(df.head())

    with st.container():
        col1, col2, col3 = st.columns(3)
        col1.metric('Rows', df.shape[0])
        col2.metric('Columns', df.shape[1])
        col3.metric('Missing Values', df.isnull().sum().sum())

    st.subheader('Statistical Summary')
    st.write(df.describe(include='all').T)

# --------------------------------------------------
# VISUALISATION
# --------------------------------------------------
elif page == 'Visualisation':
    st.header('📈 Data Visualisation')

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        feature = st.selectbox('Select numeric feature', numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax)
        st.pyplot(fig)

        st.subheader('Correlation Heatmap')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning('No numeric columns available.')

# --------------------------------------------------
# MODELING (PIPELINE)
# --------------------------------------------------
elif page == 'Modeling':
    st.header('🤖 Model Training (Pipeline)')

    target_col = st.selectbox('Select target column (Churn)', df.columns)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    st.subheader('📊 Model Performance')
    st.write(f"Accuracy: **{accuracy_score(y_test, y_pred):.3f}**")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.text(classification_report(y_test, y_pred))

    joblib.dump(pipeline, 'churn_pipeline.pkl')
    joblib.dump(target_col, 'target_col.pkl')

    st.success('✅ Pipeline trained and saved successfully!')

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
elif page == 'Prediction':
    st.header('🔮 Churn Prediction')

    try:
        pipeline = joblib.load('churn_pipeline.pkl')
        target_col = joblib.load('target_col.pkl')
    except:
        st.error('Please train the model first.')
        st.stop()

    st.markdown('### 📝 Enter Customer Details')

    input_data = {}
    col1, col2 = st.columns(2)

    for i, col in enumerate(df.drop(columns=[target_col]).columns):
        if df[col].dtype == 'object':
            if i % 2 == 0:
                with col1:
                    input_data[col] = st.selectbox(col, df[col].unique())
            else:
                with col2:
                    input_data[col] = st.selectbox(col, df[col].unique())
        else:
            if i % 2 == 0:
                with col1:
                    input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            else:
                with col2:
                    input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    input_df = pd.DataFrame([input_data])

    if st.button('🚀 Predict Churn'):
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error('⚠️ Customer is likely to churn')
            st.write(f'**Churn Probability:** {probability:.2%}')
        else:
            st.success('✅ Customer is unlikely to churn')
            st.write(f'**Churn Probability:** {probability:.2%}')
