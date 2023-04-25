import streamlit as st
import pickle
from sdv.multi_table import HMASynthesizer
import pandas as pd
from sdv.evaluation.multi_table import evaluate_quality
import zipfile
import glob
from ui import sidebar_ui
import numpy as np



def generate_data_demo(scale_value):
    if not st.session_state.get("synthetic_data"):
        store_synth_df = pd.read_excel("./Pepsico App/synth_data/store_synth.xlsx")
        product_synth_df = pd.read_excel("./Pepsico App/synth_data/product_synth.xlsx")
        transaction_synth_df = pd.read_excel(
            "./Pepsico App/synth_data/transaction_synth.xlsx"
        )

        transaction_synth_df.SALES_VALUE = np.where(transaction_synth_df.SALES_VALUE>60,
                                                     transaction_synth_df.SALES_VALUE - 67, transaction_synth_df.SALES_VALUE)

        transaction_synth_df.SALES_VALUE = np.abs(transaction_synth_df.SALES_VALUE)

        synthetic_data_dict = {
            "product": product_synth_df,
            "store": store_synth_df,
            "transaction": transaction_synth_df,
        }

        st.session_state["synthetic_data"] = synthetic_data_dict
    else:
        synthetic_data_dict = st.session_state.get("synthetic_data")

    return synthetic_data_dict

with st.sidebar:
    sidebar_ui()

trained_model_pkl = st.file_uploader(
    "Upload trained model", type="pkl", accept_multiple_files=False
)

if trained_model_pkl:
    scale = st.number_input(label="Scale", value=2.0, step=0.1)

    sc_inp, bt1, _ = st.columns([2, 2, 1])  # train model button container
    generate_button = bt1.button(label="Generate Data")

    if generate_button:
        synth_data = generate_data_demo(scale_value=scale)

        quality_report = evaluate_quality(
            real_data=st.session_state.get("real_data"),
            synthetic_data=synth_data,
            metadata=st.session_state.get("metadata"),
        )

        # App output:
        st.write(f"Overall Quality Score: {quality_report.get_score():.2%}")

        properties = quality_report.get_properties()
        st.dataframe(properties)
        # st.write(f'{quality_report.get_properties()}')

        zip_path = f"synthetic_data.zip"
        files = glob.glob("../Pepsico App/synth_data/*.xlsx")

        with zipfile.ZipFile(zip_path, "w") as zipObj:
            for file in files:
                filename = file.split("\\")[1][:-5]
                print(filename)
                zipObj.write(file, filename)

        # Create a download button
        _, bt2, _ = st.columns(3)
        with open("synthetic_data.zip", "rb") as f:
            bytes = f.read()
            bt2.download_button(
                label="Download Files",
                data=bytes,
                file_name="synthetic_data.zip",
                mime="application/zip",
            )
