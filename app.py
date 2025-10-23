# streatmlit app for insulet takehome challenge deployement
import streamlit as st
import os
import cv2
import pandas as pd
import numpy as np
import joblib
import time
from utils import create_date_variables, extract_minimal_image_features, load_images

# load models and preprocessing assets
year_min = joblib.load("models/year_min_value.joblib")
pt = joblib.load("models/power_transformer.joblib")

final_model_classifier = joblib.load("models/xgb_classifier_final.joblib")
final_model_grp0 = joblib.load("models/xgb_best_model_grp0.joblib")
final_model_grp1 = joblib.load("models/xgb_best_model_grp1.joblib")

st.markdown("<h1 style='text-align: center;'>Insulet DS takehome challenge</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Ming Yan (Oct 2025)</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload your data and get predictions using trained ML models</p>", unsafe_allow_html=True)

# upload test.csv
uploaded_file = st.file_uploader("Upload test data (.csv file)", type=["csv"])

if uploaded_file:
    df_test = pd.read_csv(uploaded_file)
    st.write("Data uploaded successfuly.")
    st.write("Preview:")
    st.dataframe(df_test)

    # a button: whether to start predicting
    if st.button("🚀 Run Prediction"):
        # transform features
        df_test['date'] = pd.to_datetime(df_test['date'])
        df_test = create_date_variables(df_test)
        df_test['year_centered'] = df_test['year'] - year_min
        df_test[['bar_transformed', 'boz_transformed']] = pt.transform(df_test[['bar', 'boz']])

        continuous_features = [
            'bar_transformed', 'boz_transformed', 'baz', 'xgt', 'lux', 'foo', 'fyt', 'lgh', 'hrt',
            'qgg', 'yyz', 'gox', 'year_centered', 'is_weekend',
            'month_sin', 'month_cos', 'dow_sin', 'dow_cos'
        ]

        features_for_grp = ['bar_transformed', 'boz_transformed', 'baz', 
                            'xgt', 'lux', 'foo', 'fyt', 'lgh', 'hrt', 
                            'qgg', 'yyz', 'gox',
                            'year_centered', 'is_weekend', 'month_sin', 
                            'month_cos', 'dow_sin', 'dow_cos']

        with st.spinner("Making predictions..."):
            # 1. first predict 0/1 group 'grp'
            df_test['grp'] = final_model_classifier.predict(df_test[features_for_grp])

            # 2. extract image features
            test_img = load_images(df_test)
            X_test_img, _ = extract_minimal_image_features(test_img)
            X_concate_test = np.concatenate([df_test[continuous_features].values, X_test_img], axis=1)

            # 3. make predictions on 'target' based on 0/1 group
            preds = np.zeros(len(df_test))
            mask0 = df_test['grp'] == 0
            mask1 = df_test['grp'] == 1
            preds[mask0] = final_model_grp0.predict(X_concate_test[mask0])
            preds[mask1] = final_model_grp1.predict(X_concate_test[mask1])

            df_test['predicted target'] = preds
            time.sleep(1)

        st.success("✅ Prediction complete!")

        # move the target_pred to the first column
        df_test = df_test.drop(columns=['grp'])
        cols = ['predicted target'] + [c for c in df_test.columns if c != 'predicted target']
        df_test = df_test[cols]
        st.dataframe(df_test)

        # if want to download results
        csv = df_test.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download predictions as CSV", data=csv, file_name="predicted_results_MingYan.csv")