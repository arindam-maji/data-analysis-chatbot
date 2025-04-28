import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from utils import (analyze_with_huggingface, generate_visualizations,
                   load_data, load_huggingface_model,
                   perform_rule_based_analysis)

# Set page configuration
st.set_page_config(
    page_title="Data Analysis Chatbot", 
    layout="wide",
    initial_sidebar_state="expanded"
)
#..
# In the analyze_clicked block, after calling perform_rule_based_analysis
# Add:
specific_calculations = perform_specific_calculations(df, query)
if specific_calculations:
    # Add to analysis_results
    analysis_results.update(specific_calculations)
# App title and description
st.title("📊 Data Analysis Chatbot")
st.write("Upload your data file and ask questions to get insights and visualizations!")

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Sidebar for file upload and options
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV, Excel, or JSON file", type=["csv", "xlsx", "json"])
    
    if uploaded_file is not None:
        # Process the uploaded file
        df, error = load_data(uploaded_file)
        
        if error:
            st.error(error)
        else:
            st.session_state['data'] = df
            st.success(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Display basic data info in sidebar
            st.subheader("Data Overview")
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
            
            # Show data types
            st.subheader("Data Types")
            dtype_counts = df.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['Data Type', 'Count']
            st.dataframe(dtype_counts)
    
    # Add model information in sidebar
    st.subheader("About")
    st.write("""
    This app uses Hugging Face models and rule-based analysis to process 
    your data queries and generate visualizations. Free and open-source!
    """)

# Main area for data preview and chat interface
if st.session_state['data'] is not None:
    df = st.session_state['data']
    
    # Data preview tab
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Load AI model (cached)
    tokenizer, model = load_huggingface_model()
    
    # Chat interface
    st.subheader("💬 Ask about your data")
    
    # Query input
    query = st.text_area(
        "Enter your question about the data:",
        placeholder="Examples:\n- Show me a summary of the data\n- What is the correlation between columns?\n- Show me distributions of numeric columns\n- Are there any missing values?",
        height=100
    )
    
    col1, col2 = st.columns([1, 5])
    analyze_clicked = col1.button("Analyze", use_container_width=True)
    clear_clicked = col2.button("Clear History", use_container_width=True)
    
    if clear_clicked:
        st.session_state['chat_history'] = []
        st.rerun()
    
    if analyze_clicked and query:
        # Add user query to chat history
        st.session_state['chat_history'].append({"role": "user", "content": query})
        
        with st.spinner("Analyzing your data..."):
            # AI Analysis with Hugging Face (if model loaded)
            if tokenizer is not None and model is not None:
                ai_response = analyze_with_huggingface(df, query, tokenizer, model)
            else:
                ai_response = "AI model not available. Using rule-based analysis only."
            
            # Rule-based analysis
            analysis_results = perform_rule_based_analysis(df, query)
            
            # Generate visualizations
            visualizations = generate_visualizations(df, query)
            
            # Combine analysis into a response
            response = f"{ai_response}\n\n"
            
            # Add response to chat history
            st.session_state['chat_history'].append({
                "role": "assistant", 
                "content": response,
                "analysis": analysis_results,
                "visualizations": visualizations
            })
    
    # Display chat history
    st.subheader("Chat History")
    for message in st.session_state['chat_history']:
        # User messages
        if message["role"] == "user":
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                <p><strong>You:</strong> {message["content"]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Assistant messages
        elif message["role"] == "assistant":
            st.markdown(f"""
            <div style='background-color: #e6f3ff; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                <p><strong>Assistant:</strong> {message["content"]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display analysis results if available
            if "analysis" in message:
                for key, value in message["analysis"].items():
                    if key == "summary":
                        st.subheader("Statistical Summary")
                        st.dataframe(value)
                    
                    elif key == "correlation":
                        st.subheader("Correlation Matrix")
                        st.dataframe(value)
                    
                    elif key == "missing":
                        st.subheader("Missing Values")
                        if isinstance(value, str):
                            st.write(value)
                        else:
                            st.dataframe(value)
                    
                    elif key == "unique_counts":
                        st.subheader("Unique Value Counts")
                        st.json(value)
                    
                    elif key == "column_info":
                        st.subheader("Column Information")
                        st.write("Data Types:")
                        st.write(value["dtypes"])
                        st.write("Non-null Counts:")
                        st.write(value["non_nulls"])
                        st.write("Unique Value Counts:")
                        st.json(value["unique_counts"])
                    
                    elif key == "basic_info":
                        st.subheader("Basic Data Information")
                        st.write(f"Shape: {value['shape']}")
                        st.write(f"Columns: {value['columns']}")
                        st.write(f"Total missing values: {value['missing_total']}")
            
            # Display visualizations if available
            if "visualizations" in message and message["visualizations"]:
                st.subheader("Visualizations")
                for viz_name, fig in message["visualizations"].items():
                    st.pyplot(fig)
else:
    # Show welcome message when no data is loaded
    st.info("👈 Please upload a data file in the sidebar to get started!")
    
    # Example capabilities
    st.subheader("What can this chatbot do?")
    st.markdown("""
    - **Upload various data formats** (CSV, Excel, JSON)
    - **Ask questions** about your data in natural language
    - **Get statistical insights** like summaries, correlations, and distributions
    - **Visualize your data** with automatic charts and plots
    - **Identify patterns and issues** like missing values or outliers
    
    Upload a file to get started!
    """)