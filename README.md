# Data Analysis Chatbot

A user-friendly data analysis tool built with Streamlit and Hugging Face that allows users to upload data files and analyze them using natural language queries.

## Features

- Upload CSV, Excel, or JSON data files
- Ask questions about your data in plain English
- Get AI-powered insights from Hugging Face models
- View automatic visualizations and statistical analysis
- No cost - built with free and open-source technologies

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/data-analysis-chatbot.git
cd data-analysis-chatbot
```

2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run the application:
```
streamlit run app.py
```

## Usage

1. Upload your data file using the sidebar
2. Ask questions about your data in the text area
3. Click "Analyze" to get insights and visualizations
4. Review the chat history for previous analyses

## Example Queries

- "Show me a summary of the data"
- "What are the correlations between numeric columns?"
- "Show me distributions of numeric columns"
- "Are there any missing values in the dataset?"
- "Show me counts of categorical variables"

## Deployment

This application can be deployed for free on Streamlit Cloud:
1. Push this code to a GitHub repository
2. Sign up for a free Streamlit Cloud account
3. Connect your GitHub repository
4. Deploy the app

## Built With

- [Streamlit](https://streamlit.io/) - The web framework
- [Hugging Face Transformers](https://huggingface.co/transformers/) - AI models
- [Pandas](https://pandas.pydata.org/) - Data analysis
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Visualizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.