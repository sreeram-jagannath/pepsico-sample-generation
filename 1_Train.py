import streamlit as st
from sdv.metadata import MultiTableMetadata
import random
import pandas as pd
import os
import json
import xmltodict
import pickle
import time


SEED = random.seed(9001)


def train_model():

    with open("./Pepsico App/model/SDVv1.0_Dunhumby_0.01.pkl", 'r') as f:
        model = pickle.load(f)

    return model


if __name__ == "__main__":
    metadata_file, hh_demograph_file, product_file, transaction_file = st.file_uploader(
        label="Upload real data",
        accept_multiple_files=True,
    )

    if metadata_file:
        hh_demographic_df = pd.read_csv(hh_demograph_file)

        # get unique values from the required columns
        AGE_DESC_options = hh_demographic_df["AGE_DESC"].unique().tolist()
        HOMEOWNER_DESC_options = hh_demographic_df["HOMEOWNER_DESC"].unique().tolist()
        HH_COMP_DESC_options = hh_demographic_df["HH_COMP_DESC"].unique().tolist()

        # create filters for the three columns
        filter1, filter2, filter3 = st.columns(3)
        AGE_DESC_filter = filter1.multiselect(
            label="AGE DESC", options=AGE_DESC_options
        )
        HOMEOWNER_DESC_filter = filter2.multiselect(
            label="HOMEOWNER_DESC", options=HOMEOWNER_DESC_options
        )
        HH_COMP_DESC_filter = filter3.multiselect(
            label="HH_COMP_DESC", options=HH_COMP_DESC_options
        )

        hh_demographic_frac = st.slider(
            label="Sampling Fraction",
            min_value=0.00,
            max_value=1.00,
            value=1.00,
            step=0.01,
        )

        hh_demographic_df = hh_demographic_df.sample(
            frac=hh_demographic_frac, random_state=SEED
        )
        hh_demographic_df = hh_demographic_df[
            hh_demographic_df["AGE_DESC"].isin(AGE_DESC_filter)
            & hh_demographic_df["HOMEOWNER_DESC"].isin(HOMEOWNER_DESC_filter)
            & hh_demographic_df["HH_COMP_DESC"].isin(HH_COMP_DESC_filter)
        ]

        # read the transaction data
        transaction_df = pd.read_csv(transaction_file)
        transaction_df = transaction_df[
            [
                "household_key",
                "BASKET_ID",
                "PRODUCT_ID",
                "QUANTITY",
                "SALES_VALUE",
                "TRANS_TIME",
            ]
        ]
        transaction_df = transaction_df[
            transaction_df["household_key"].isin(hh_demographic_df.household_key)
        ]

        # read the product data
        product_df = pd.read_csv(product_file)
        product_df = product_df[['PRODUCT_ID',	'DEPARTMENT',	'BRAND']]
        product_df =  product_df[product_df['PRODUCT_ID'].isin(transaction_df.PRODUCT_ID)]

        # 1. Collect all the tables in a dict
        real_data = {}
        real_data["product"] = product_df
        real_data["transaction"] = transaction_df
        real_data["hh_demographic"] = hh_demographic_df

        # read in the metadata
        # json_metadata = json.loads(json.dumps(xmltodict.parse(metadata_file)))
        # metadata = MultiTableMetadata.load_from_json(json_metadata)

        

        train_button = st.button(label="Train Model")
        if train_button:
            model = train_model()

            model_save_filename = f'SDVv1.0_Dunhumby_{hh_demographic_frac}.pkl'
            st.download_button(label="Download model", data=model, file_name=model_save_filename)
