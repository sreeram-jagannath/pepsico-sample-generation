import streamlit as st
from sdv.evaluation.multi_table import get_column_plot
from ui import sidebar_ui


options = {
    "store": ["STORE_TYPE", "REGION"],
    "transaction": ["QUANTITY",	"SALES_VALUE",	"TRANS_TIME",],
    "product": ["DEPARTMENT", "BRAND"],
}

with st.sidebar:
    sidebar_ui()


table_select = st.selectbox(label="Select Table", options=options.keys(), index=0)
column_select = st.selectbox(
    label="Select Column", options=options.get(table_select), index=0
)

if st.session_state.get("synthetic_data"):
    fig = get_column_plot(
        real_data=st.session_state.get("real_data"),
        synthetic_data=st.session_state.get("synthetic_data"),
        table_name=table_select,
        column_name=column_select,
        metadata=st.session_state.get("metadata"),
    )

    st.title("")

    st.plotly_chart(fig)
