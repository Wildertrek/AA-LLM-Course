import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def app():
    st.title("Welcome to Page 1")

    st.write("""
    ## Data Visualization Dashboard
    This page demonstrates basic data visualization capabilities.
    Upload your CSV file or use our sample data to generate insights.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Data source selection
    use_sample_data = st.checkbox("Use sample data instead", value=True if uploaded_file is None else False)

    # Load data
    if uploaded_file is not None and not use_sample_data:
        df = pd.read_csv(uploaded_file)
        st.success("Your data has been loaded successfully!")
    else:
        # Sample data
        st.info("Using sample data")
        df = pd.DataFrame({
            'Category': ['A', 'B', 'C', 'D', 'E'],
            'Values': [23, 42, 15, 30, 25]
        })

    # Display dataframe
    st.subheader("Data Preview")
    st.dataframe(df)

    # Simple visualization
    st.subheader("Data Visualization")

    chart_type = st.selectbox("Select chart type", ["Bar Chart", "Line Chart", "Pie Chart"])

    if chart_type == "Bar Chart":
        fig, ax = plt.subplots()
        ax.bar(df.iloc[:, 0], df.iloc[:, 1])
        st.pyplot(fig)
    elif chart_type == "Line Chart":
        st.line_chart(df.set_index(df.columns[0]))
    elif chart_type == "Pie Chart":
        fig, ax = plt.subplots()
        ax.pie(df.iloc[:, 1], labels=df.iloc[:, 0], autopct='%1.1f%%')
        st.pyplot(fig)

    st.info("This is a basic template. Customize it according to your needs!")

if __name__ == "__main__":
    app()