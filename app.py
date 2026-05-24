import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.io as pio
pio.templates.default = "plotly_dark"

# Page Configuration

st.set_page_config(
    page_title="Financial Complaint Intelligence System",
    layout="wide"
)

# Styling the dashboard

st.markdown("""
<style>

// Main App Background


.stApp {
    background:
    radial-gradient(circle at top left, #172554 0%, transparent 30%),
    radial-gradient(circle at top right, #1E3A8A 0%, transparent 25%),
    linear-gradient(135deg, #020617, #050816, #0F172A);

    color: #F8FAFC;
}
            
// Remove Excess Top Space 

.block-container {

    padding-top: 0.2rem;

    padding-bottom: 2rem;

    padding-left: 2rem;

    padding-right: 2rem;
}

// Force Content To Start From Top 

.main .block-container {

    max-width: 100%;

    padding-top: 0.8rem;

    margin-top: 0rem;
}

// Global Text


html, body, [class*="css"] {
    color: #F8FAFC;
}

// Sidebar section

section[data-testid="stSidebar"] {

    background: linear-gradient(
        180deg,
        #050B18,
        #0B1120
    );

    border-right: 1px solid #1E293B;
}

// Sidebar Labels

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #F8FAFC !important;
}

// Inputs

/ Multiselect

.stMultiSelect div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    color: white !important;
    border: 1px solid #2563EB;
    border-radius: 14px;
}

/ Selectbox

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    color: white !important;
    border: 1px solid #2563EB;
    border-radius: 14px;
}

/ Search Input

.stTextInput input {
    background-color: #111827 !important;
    color: white !important;
    border: 1px solid #3B82F6 !important;
    border-radius: 14px;
    padding: 12px;
}

/ Placeholder

::placeholder {
    color: #CBD5E1 !important;
    opacity: 1;
}

/ Headings

h1, h2, h3 {

    color: #FFFFFF !important;

    font-weight: 800;

    letter-spacing: 0.4px;
}

/ Metric Cards

[data-testid="metric-container"] {

    background:
    linear-gradient(
        145deg,
        rgba(15,23,42,0.92),
        rgba(30,41,59,0.82)
    );

    border: 1px solid rgba(59,130,246,0.25);

    padding: 22px;

    border-radius: 18px;

    box-shadow:
    0 8px 32px rgba(0,0,0,0.35);

    backdrop-filter: blur(12px);

    transition: all 0.3s ease;
}

// Hover 

[data-testid="metric-container"]:hover {

    transform: translateY(-5px);

    border: 1px solid #3B82F6;

    box-shadow:
    0 0 20px rgba(59,130,246,0.25);
}

// KPI Labels 

[data-testid="stMetricLabel"] {

    color: #CBD5E1 !important;

    font-size: 16px !important;

    font-weight: 600 !important;
}

// KPI Values 

[data-testid="stMetricValue"] {

    color: #FFFFFF !important;

    font-size: 38px !important;

    font-weight: 800 !important;
}

// Tabs

.stTabs [data-baseweb="tab-list"] {
    gap: 28px;
}

.stTabs [data-baseweb="tab"] {

    color: #CBD5E1;

    font-weight: 600;

    font-size: 16px;

    transition: 0.3s ease;
}

// Active Tab

.stTabs [aria-selected="true"] {

    color: #60A5FA !important;
}

// Active Tab Border

button[aria-selected="true"] {

    border-bottom: 2px solid #3B82F6 !important;
}

// Dataframe

[data-testid="stDataFrame"] {

    border-radius: 16px;

    overflow: hidden;

    border: 1px solid #1E293B;

    color: white !important;
}

// Download Button
 

.stDownloadButton button {

    background: linear-gradient(
        135deg,
        #2563EB,
        #1D4ED8
    );

    color: white;

    border-radius: 14px;

    border: none;

    padding: 0.7rem 1.4rem;

    font-weight: 600;

    transition: 0.3s ease;
}

// Button Hover 

.stDownloadButton button:hover {

    transform: translateY(-2px);

    box-shadow:
    0 0 18px rgba(59,130,246,0.4);
}

// Alert Cards

// Success

[data-testid="stSuccess"] {

    background:
    linear-gradient(
        135deg,
        rgba(16,185,129,0.18),
        rgba(5,150,105,0.08)
    );

    border: 1px solid rgba(16,185,129,0.45);

    color: #D1FAE5;

    border-radius: 16px;

    padding: 18px;
}

// Warning

[data-testid="stWarning"] {

    background:
    linear-gradient(
        135deg,
        rgba(245,158,11,0.18),
        rgba(180,83,9,0.08)
    );

    border: 1px solid rgba(245,158,11,0.45);

    color: #FEF3C7;

    border-radius: 16px;

    padding: 18px;
}

// Error

[data-testid="stError"] {

    background:
    linear-gradient(
        135deg,
        rgba(239,68,68,0.18),
        rgba(127,29,29,0.08)
    );

    border: 1px solid rgba(239,68,68,0.45);

    color: #FEE2E2;

    border-radius: 16px;

    padding: 18px;
}

// Main Content

.block-container {

    padding-top: 2rem;

    padding-bottom: 2rem;

    padding-left: 3rem;

    padding-right: 3rem;
}

// Charts

.element-container {

    border-radius: 18px;
}

// General Labels

label, .stMarkdown, p {
    color: #E5E7EB !important;
}

</style>
""", unsafe_allow_html=True)

# Load Data

df = pd.read_csv("dashboard_ready_dataset.csv")


# Product Name Cleaning

product_mapping = {
    'checking or savings account': 'checking/savings',
    'credit card or prepaid card': 'credit card',
    'credit reporting, credit repair services, or other personal consumer reports': 'credit reporting',
    'money transfer, virtual currency, or money service': 'money transfer',
    'vehicle loan or lease': 'vehicle loan',
    'payday loan, title loan, or personal loan': 'payday/personal loan'
}

df['product_short'] = (
    df['product']
    .replace(product_mapping)
)

df['issue_short'] = (
    df['issue']
    .str.slice(0, 35)
)


# Title Section


st.title("Financial Complaint Intelligence System")

st.markdown("""
AI-powered operational analytics platform designed to monitor customer complaints,
identify operational inefficiencies, analyze service risks,
and generate actionable business intelligence insights.
""")

st.markdown("---")

# Sidebar Filters

st.sidebar.header("Dashboard Filters")

selected_products = st.sidebar.multiselect(
    "Select Product",
    options=sorted(df['product_short'].unique()),
    default=sorted(df['product_short'].unique())
)

selected_states = st.sidebar.multiselect(
    "Select State",
    options=sorted(df['state'].unique()),
    default=sorted(df['state'].unique())
)

selected_response_speed = st.sidebar.multiselect(
    "Response Speed",
    options=sorted(df['response_speed_category'].unique()),
    default=sorted(df['response_speed_category'].unique())
)

# applying sidebar selections dynamically so all KPIs and charts update together

filtered_df = df[
    (df['product_short'].isin(selected_products)) &
    (df['state'].isin(selected_states)) &
    (df['response_speed_category'].isin(selected_response_speed))
]

# stopping dashboard execution if selected filters return no matching complaints

if filtered_df.empty:

    st.error("No data available for selected filters.")

    st.stop()

# Global Analytics Variables

# yields product with highest number of complaints
highest_complaint_product = (
    filtered_df['product_short']
    .value_counts()
    .idxmax()
)

# yields product category which has the highest average response delay
highest_delay_product = (
    filtered_df
    .groupby('product_short')['response_days']
    .mean()
    .idxmax()
)

# yields the value of highest average response delay days
highest_delay_days = round(
    filtered_df
    .groupby('product_short')['response_days']
    .mean()
    .max(),
    2
)

# Tabs


tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Overview",
    "Operational Analytics",
    "NLP Intelligence",
    "Complaint Explorer"
])

# Tab 1 — Executive Overview

with tab1:

    st.subheader("Executive Overview")

    total_complaints = len(filtered_df)

    avg_response_days = round(
        filtered_df['response_days'].mean(),
        2
    )

    timely_response_rate = round(
        filtered_df['timely_binary'].mean() * 100,
        2
    )

    most_complained_product = (
        filtered_df['product_short']
        .value_counts()
        .idxmax()
    )

    high_severity_count = (
        filtered_df['severity_level'] == 'high'
    ).sum()

    # KPI cards

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Complaints",
        f"{total_complaints:,}"
    )

    col2.metric(
        "Avg Response Days",
        avg_response_days
    )

    col3.metric(
        "Timely Response %",
        f"{timely_response_rate}%"
    )

    col4.metric(
        "Most Complained Product",
        most_complained_product
    )

    col5.metric(
        "High Severity Complaints",
        f"{high_severity_count:,}"
    )

    st.markdown("---")

    # Operational Alerts

    st.subheader("Operational Alerts")

    st.warning(
        f"""
        High complaint concentration detected in
        {highest_complaint_product}.
        """
    )

    st.error(
        f"""
        {highest_delay_product}
        shows the highest average response delay
        ({highest_delay_days} days).
        """
    )

    st.markdown("---")

    # Recommendations

    st.subheader("Strategic Recommendations")

    if highest_delay_days > 7:

        st.warning(
            f"""
            Recommendation:

            Improve complaint resolution workflows,
            escalation handling,
            and operational responsiveness
            for {highest_delay_product} complaints.
            """
        )

    else:

        st.success(
            """
            Complaint response performance
            remains within acceptable thresholds.
            """
        )

# Tab 2 — Operational Analytics

with tab2:

    # Monthly Trend

    st.subheader("Monthly Complaint Trend")

    monthly_trend = (
        filtered_df
        .groupby('year_month')
        .size()
        .reset_index(name='complaint_count')
    )

    fig_trend = px.line(
        monthly_trend,
        x='year_month',
        y='complaint_count'
    )

    fig_trend.update_layout(
        xaxis_title='Timeline',
        yaxis_title='Complaint Count',
        height=450
    )

    st.plotly_chart(
        fig_trend,
        use_container_width=True
    )

    # Top Complaint Products plot

    st.subheader("Top Complaint Products")

    product_counts = (
        filtered_df['product_short']
        .value_counts()
        .head(10)
        .reset_index()
    )

    product_counts.columns = [
        'product',
        'complaint_count'
    ]

    fig_products = px.bar(
        product_counts,
        x='product',
        y='complaint_count'
    )

    fig_products.update_layout(
        xaxis_title='Product',
        yaxis_title='Complaint Count',
        height=450
    )

    st.plotly_chart(
        fig_products,
        use_container_width=True
    )

    # Response delays by product plot

    st.subheader("Average Response Delay by Product")

    delay_analysis = (
        filtered_df
        .groupby('product_short')['response_days']
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    delay_analysis.columns = [
        'product',
        'avg_response_days'
    ]

    fig_delay = px.bar(
        delay_analysis,
        x='product',
        y='avg_response_days'
    )

    fig_delay.update_layout(
        xaxis_title='Product',
        yaxis_title='Average Response Days',
        height=450
    )

    st.plotly_chart(
        fig_delay,
        use_container_width=True
    )

# Tab 3 — NLP Intelligence

with tab3:

    st.subheader("NLP Complaint Intelligence")


    # TF-IDF Analysis

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=20
    )

    X = vectorizer.fit_transform(
        filtered_df['cleaned_text']
    )

    keywords = vectorizer.get_feature_names_out()

    scores = X.sum(axis=0).A1

    keyword_df = pd.DataFrame({
        'keyword': keywords,
        'importance': scores
    })

    keyword_df = keyword_df.sort_values(
        by='importance',
        ascending=False
    )

    # Keyword Chart on the basis of it's word importance done by TF IDF

    st.subheader("Top Complaint Keywords")

    fig_keywords = px.bar(
    keyword_df.head(10),
    x='keyword',
    y='importance',
    text_auto='.2s'
)

    fig_keywords.update_layout(
    xaxis_title='Keyword',
    yaxis_title='Importance Score',
    height=400,
    showlegend=False
)

    fig_keywords.update_traces(
    textposition='outside'
)

    fig_keywords.update_layout(
        xaxis_title='Keyword',
        yaxis_title='Importance Score',
        height=450
    )

    st.plotly_chart(
        fig_keywords,
        use_container_width=True
    )

    # Word Cloud image of the most important keywords

    st.subheader("Complaint Word Cloud")

    all_text = " ".join(
        filtered_df['cleaned_text'].astype(str)
    )

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color='white'
    ).generate(all_text)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.imshow(wordcloud, interpolation='bilinear')

    ax.axis('off')

    st.pyplot(fig)

    # Issue Themes and frequency plot

    st.subheader("Most Frequent Complaint Issues")

    issue_counts = (
        filtered_df['issue_short']
        .value_counts()
        .head(10)
        .reset_index()
    )

    issue_counts.columns = [
        'issue',
        'count'
    ]

    fig_issues = px.bar(
    issue_counts,
    x='issue',
    y='count',
    text_auto=True
)

    fig_issues.update_layout(
    xaxis_title='Issue Type',
    yaxis_title='Complaint Count',
    height=450,
    showlegend=False
)

    fig_issues.update_traces(
    textposition='outside'
)
    
    col1, col2, col3 = st.columns(3)

    col1.metric(
    "Top Keyword",
    keyword_df.iloc[0]['keyword']
)

    col2.metric(
    "Top Complaint Theme",
    issue_counts.iloc[0]['issue']
)

    col3.metric(
    "Keyword Count",
    len(keyword_df)
)

    # NLP Insight after going through most important keyword and the frequency of the keyword

    top_keyword = keyword_df.iloc[0]['keyword']

    top_issue = issue_counts.iloc[0]['issue']

    st.success(
        f"""
        NLP Insight:

        Dominant complaint language is strongly associated with
        '{top_keyword}' related concerns,
        while the most frequent issue category is
        '{top_issue}'.
        """
    )

# TAB 4 — Complaint Explorer 

with tab4:



    st.subheader("Interactive Complaint Explorer")

    # Search Bar

    search_keyword = st.text_input(
        "Search Complaint Narratives",
        placeholder="Type keywords like fraud, payment, credit, delay..."
    )

    # Issue Filter

    selected_issue = st.selectbox(
        "Select Complaint Issue",
        sorted(filtered_df['issue'].unique())
    )

    # Filter Data

    explorer_df = filtered_df[
        filtered_df['issue'] == selected_issue
    ]

    # Search Logic
    

    if search_keyword:

        explorer_df = explorer_df[
            explorer_df['complaint_narrative']
            .str.contains(
                search_keyword,
                case=False,
                na=False
            )
        ]

    # Results Count

    st.markdown(
        f"### Matching Complaints: {len(explorer_df)}"
    )

    # Display Data

    st.dataframe(
        explorer_df[
            [
                'product_short',
                'state',
                'issue',
                'response_days',
                'complaint_narrative'
            ]
        ],
        use_container_width=True,
        height=500
    )

        
    # Download Report Feature
    

    csv = explorer_df.to_csv(index=False)

    st.download_button(
        label="Download Filtered Complaint Report",
        data=csv,
        file_name="filtered_complaints_report.csv",
        mime="text/csv"
    )
# Optional Data Preview

with st.expander("View Dataset Preview"):

    st.dataframe(
        filtered_df.head(20),
        use_container_width=True
    )