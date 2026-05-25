## Financial Complaint Intelligence System

An AI-powered Financial Complaint Intelligence System built using Python, Streamlit, NLP, and Tableau to analyze consumer financial complaints, identify operational inefficiencies, monitor response delays, and generate actionable business insights through interactive analytics and natural language processing.

# Live Deployed Link: [https://anshika15p-financial-complaint-intelligence-system-app-zynveg.streamlit.app/]

**Project Motivation**

Financial institutions receive large volumes of customer complaints related to products, services, delayed responses, and operational inefficiencies. The objective of this project was to analyze complaint data from a business intelligence perspective and understand:
- Which financial products receive the highest complaint volume?
- Which operational areas experience delayed responses?
- How complaint patterns vary across states and time periods?
- What common themes and keywords dominate customer complaints?
- How NLP can help extract meaningful insights from unstructured complaint narratives?
The project combines exploratory data analysis, NLP-based complaint intelligence, interactive Streamlit analytics, and Tableau business storytelling dashboards.

**Project Workflow**

1. *Data Cleaning & Preprocessing*
   
The project began with extensive preprocessing and exploratory data analysis to prepare the dataset for analytical workflows.
Key preprocessing tasks included:
- Handling missing and duplicate values
- Standardizing categorical fields
- Creating engineered features for dashboard analytics
- Analyzing complaint distribution across products, states, submission channels, and response categories
- Additional engineered fields were created to support operational analytics and dashboard filtering.
  
2. *Exploratory Data Analysis (EDA)*
   
EDA was performed to understand complaint behavior and operational patterns within the dataset.
The analysis focused on:
- Complaint distribution by product category
- State-wise complaint concentration
- Complaint submission channels
- Monthly complaint trends
- Average response delay analysis
- Timely vs delayed response patterns
These insights helped establish the business intelligence foundation of the project.

3. *NLP-Based Complaint Intelligence*

To enhance the NLP analysis, synthetic complaint narratives were generated using template-based narrative augmentation to simulate customer complaint messages across the dataset.
The NLP pipeline included:
- Text cleaning and preprocessing
- Stopword removal
- Tokenization
- TF-IDF vectorization
  
**Keyword importance analysis**
**TF-IDF (Term Frequency–Inverse Document Frequency)** was used to identify the most informative and operationally significant complaint-related terms.
This helped uncover:
- Dominant complaint themes
- Frequently occurring customer issues
- High-impact operational keywords
- Complaint language patterns
  
The results were **visualized** using:
- Top TF-IDF keyword charts
- Complaint theme analysis
- Word cloud visualizations
  
**Streamlit Dashboard Features**
The project was deployed as an interactive Streamlit analytical application with four primary modules:
- *Executive Overview*
Provides high-level operational KPIs including:
    - Total complaints
    - Average response days
    - Timely response percentage
    - Most complained product
    - Operational alerts
    - Strategic recommendations
  
- *Operational Analytics*
Focused on operational monitoring and trend analysis:
    - Monthly complaint trends
    - Top complaint products
    - Average response delay by product category
    - NLP Complaint Intelligence
      
- *Provides NLP-driven insights using TF-IDF analysis*:
    - Top complaint keywords
    - Complaint word cloud
    - Complaint theme analysis
    - Keyword frequency intelligence
      
- *Complaint Explorer*
An interactive complaint search engine allowing users to:
    - Search complaint narratives using keywords
    - Filter complaints by issue category
    - Explore matching complaint records dynamically
    - Download filtered complaint reports
This feature was designed to improve complaint investigation workflows and make large complaint datasets easier to analyze interactively.

**Technologies Used**
- Python
- Pandas
- NumPy
- Streamlit
- Plotly
- Scikit-learn
- NLP / TF-IDF
- WordCloud
- Tableau
- Deployment

**Streamlit Application**:
- Live Dashboard
  
**Future Improvements**
- Real-time complaint ingestion
- Predictive complaint escalation modeling
- LLM-based complaint summarization
- Automated operational risk scoring
  

