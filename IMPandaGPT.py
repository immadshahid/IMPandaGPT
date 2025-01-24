import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from langchain_experimental.agents import create_csv_agent
from langchain.agents import create_sql_agent
from langchain.chat_models import AzureChatOpenAI
import matplotlib.pyplot as plt
import seaborn as sns

# Azure OpenAI Configuration
os.environ["OPENAI_API_KEY"] = "Add your OPENAI API Key"
os.environ["OPENAI_API_BASE"] = "Add endpoint or base"
os.environ["OPENAI_API_TYPE"] = "add type"
os.environ["OPENAI_API_VERSION"] = "Add Version"

openai_model = AzureChatOpenAI(
    deployment_name="Write you model name",
    temperature=0.9
)

st.title("IMPandaGPT")
st.sidebar.header("Data Selection")

# Data Source Selection
data_source = st.sidebar.selectbox("Select a Data Source", ["CSV", "SQL"])

uploaded_csv = None
csv_data = None
query_result = None

# Upload CSV
if data_source == "CSV":
    st.subheader("Upload a CSV File")
    uploaded_csv = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_csv is not None:
        csv_data = pd.read_csv(uploaded_csv)
        st.write("Preview of Uploaded CSV:")
        st.write(csv_data)

        # Save the CSV locally for the agent
        csv_file_path = "uploaded_file.csv"
        csv_data.to_csv(csv_file_path, index=False)

# Initialize Agents
csv_agent = None
sql_agent = None

if uploaded_csv is not None:
    # Create a CSV Agent
    csv_agent = create_csv_agent(
        llm=openai_model,
        path=csv_file_path,
        verbose=True,
        allow_dangerous_code=True
    )

# Handle SQL data
if data_source == "SQL" and uploaded_csv is not None:
    # Save CSV to SQLite
    engine = create_engine("sqlite:///uploaded_data.db")
    csv_data.to_sql("uploaded_data", engine, if_exists="replace", index=False)

    # Create an SQL Agent
    sql_agent = create_sql_agent(
        llm=openai_model,
        toolkit=engine,
        verbose=False
    )

# Data Visualization Function
def plot_visualization(data, plot_type, x_col=None, y_col=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    if plot_type == "Bar Chart":
        sns.barplot(x=data[x_col], y=data[y_col], ax=ax)
    elif plot_type == "Line Chart":
        sns.lineplot(x=data[x_col], y=data[y_col], ax=ax)
    elif plot_type == "Histogram":
        sns.histplot(data[x_col], kde=True, ax=ax)
    elif plot_type == "Scatter Plot":
        sns.scatterplot(x=data[x_col], y=data[y_col], ax=ax)
    st.pyplot(fig)

# Interact with Dataset
if uploaded_csv is not None:
    st.subheader("Interact with Your Dataset")

    if data_source == "CSV" and csv_agent:
        st.markdown("**Let's chat about your CSV data!**")
        query = st.text_input("Ask a question about your CSV data:")
        if st.button("Submit Query"):
            if query.strip():
                with st.spinner("Processing your query..."):
                    query_result = csv_agent.run(query)
                    st.success("Query completed!")
                    st.write("Query Result:")
                    st.write(query_result)

        # Visualization Options
        st.subheader("Visualize Your Data")
        visualization_type = st.selectbox(
            "Select Visualization Type",
            ["None", "Bar Chart", "Line Chart", "Histogram", "Scatter Plot"]
        )

        if visualization_type != "None":
            numeric_columns = csv_data.select_dtypes(include=["float64", "int64"]).columns.tolist()
            all_columns = csv_data.columns.tolist()

            if visualization_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
                x_axis = st.selectbox("Select X-axis", options=all_columns)
                y_axis = st.selectbox("Select Y-axis", options=numeric_columns)
                if st.button("Generate Visualization"):
                    plot_visualization(csv_data, visualization_type, x_axis, y_axis)

            elif visualization_type == "Histogram":
                column = st.selectbox("Select Column for Histogram", options=numeric_columns)
                if st.button("Generate Histogram"):
                    plot_visualization(csv_data, visualization_type, x_col=column)

    elif data_source == "SQL" and sql_agent:
        st.markdown("**Let's chat about your SQL data!**")
        query = st.text_input("Ask a question about your SQL database:")
        if st.button("Submit Query"):
            if query.strip():
                with st.spinner("Processing your query..."):
                    query_result = sql_agent.run(query)
                    st.success("Query completed!")
                    st.write("Query Result:")
                    st.write(query_result)


st.sidebar.markdown("---")
st.sidebar.markdown("Developed with 💙 using LangChain and Azure OpenAI.")
