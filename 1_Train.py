import streamlit as st
from sdv.metadata import MultiTableMetadata
import random
import pandas as pd
import os
import json
import pickle
import time
import io
from sdv.multi_table import HMASynthesizer
import base64


SEED = random.seed(9001)


def train_model_demo(metadata, real_data):
    with open("./Pepsico App/model/SDVv1.0_Dunhumby_0.01.pkl", "rb") as f:
        model = pickle.load(f)

    pkl_model = io.BytesIO()
    pickle.dump(model, pkl_model)

    return pkl_model


def train_model_real(metadata, real_data):
    # Step 1: Create the synthesizer
    synthesizer = HMASynthesizer(metadata)

    # Step 2: Train the synthesizer
    synthesizer.fit(real_data)

    # save the pickle model
    pkl_model = io.BytesIO()
    pickle.dump(synthesizer, pkl_model)

    return pkl_model


def sidebar_ui():
    img = "./pepsico-logo.png"
    st.markdown(
         f"""
        <style>
            <center><img  src="data:image/png;base64,{base64.b64encode(open(img, "rb").read()).decode()}"></center>
            
        </style>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    with st.sidebar:
        sidebar_ui()
    st.header("Upload files")

    fu1, fu2 = st.columns(2)
    fu3, fu4 = st.columns(2)

    metadata_file = fu1.file_uploader(label="Metadata", type="json")
    store_file = fu2.file_uploader(label="Store data", type="csv")
    product_file = fu3.file_uploader(label="Product file", type="csv")
    transaction_file = fu4.file_uploader(label="Transaction file", type="csv")

    st.divider()

    if metadata_file and store_file and product_file and transaction_file:
        store_df = pd.read_csv("./Pepsico App/real_data/store.csv")

        # get unique values from the required columns
        REGION_options = store_df["REGION"].unique().tolist()
        STORE_TYPE_options = store_df["STORE_TYPE"].unique().tolist()

        # create filters for the two columns
        filter1, filter2 = st.columns(2)
        REGION_filter = filter1.multiselect(
            label="REGION",
            options=REGION_options,
        )
        STORE_TYPE_filter = filter2.multiselect(
            label="STORE TYPE",
            options=STORE_TYPE_options,
        )

        store_frac = st.slider(
            label="Sampling Fraction",
            min_value=0.00,
            max_value=1.00,
            value=0.10,
            step=0.01,
        )

        st.divider()

        store_df = store_df[
            store_df["STORE_TYPE"].isin(STORE_TYPE_filter)
            & store_df["REGION"].isin(REGION_filter)
        ]
        store_df = store_df.sample(frac=store_frac, random_state=SEED)

        # read the transaction data
        transaction_df = pd.read_csv("./Pepsico App/real_data/transactions_store.csv")
        transaction_df = transaction_df[
            transaction_df["STORE_ID"].isin(store_df.STORE_ID)
        ]
        transaction_df = transaction_df.sample(frac=0.1)

        # read the product data
        product_df = pd.read_csv("./Pepsico App/real_data/product.csv")
        product_df = product_df[["PRODUCT_ID", "DEPARTMENT", "BRAND"]]
        product_df = product_df[
            product_df["PRODUCT_ID"].isin(transaction_df.PRODUCT_ID)
        ]

        # 1. Collect all the tables in a dict
        real_data = {}
        real_data["product"] = product_df
        real_data["transaction"] = transaction_df
        real_data["store"] = store_df

        # read in the metadata
        metadata = MultiTableMetadata.load_from_json(
            filepath="./Pepsico App/real_data/Dunhumby_Metadata_v2.json"
        )

        # store the real data in session state, so that we can use them
        # from different tabs of the web app
        # if not st.session_state.get("real_data"):
        st.session_state["real_data"] = real_data
        st.session_state["metadata"] = metadata

        st.write(st.session_state.get("real_data"))

        _, bt1, _ = st.columns([2, 2, 1])  # train model button container
        _, bt2, _ = st.columns([1.7, 2, 1])  # download model button container
        train_button = bt1.button(label="Train Model")
        if train_button:
            pkl_model = train_model_demo(metadata="", real_data=real_data)

            model_save_filename = f"SDVv1.0_Dunhumby_{store_frac}.pkl"
            bt2.download_button(
                label="Download .pkl model",
                data=pkl_model,
                file_name=model_save_filename,
            )
