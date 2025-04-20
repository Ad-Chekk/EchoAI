import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import time
import numpy as np
from datetime import datetime

# Set page config for wide layout and dark theme
st.set_page_config(layout="wide", page_title="Web Scraper Analytics", page_icon="🌐")

# Custom CSS for dark theme and styling
st.markdown("""
<style>
     [data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1603366615917-1fa6dad5c4fa?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size: cover;
}       
    .main {
       
        color: white;
    }
    .stPlotlyChart {
        background-color: transparent !important;
    }
    h1, h2, h3 {
        color: white;
    }
    
    .metric-card {
    background-color: rgba(31, 41, 55, 0.5);  /* More transparent background (0.5 opacity) */
    border-radius: 10px;
    padding: 0px;  /* Reduced padding from 20px to 15px */
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 0;
    height: 110px;
    }
    .stExpander {
        background-color: #1f2937 !important;
    }
</style>
""", unsafe_allow_html=True)

# App title and intro
st.title("🌐 Web Scraper Analytics Dashboard")
st.markdown("Visualizing the metadata collected from your web scraping activities")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("scraper_metadata.csv")
    # Extract domain from URL
    df["Domain"] = df["URL"].str.extract(r'https?://([^/]+)')
    # Convert response time to ms for better display
    df["Response Time (ms)"] = df["Response Time (s)"] * 1000
    return df

df = load_data()

# Key metrics in a row
# Key metrics in a row
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container():
        st.markdown("""
             <div class='metric-card'>
                <h3>Total Sites</h3>
                <h2 style="margin:0; font-size:2.0rem;">{}</h2>
            </div>
        """.format(len(df)), unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown("""
            <div class='metric-card'>
                <h3>Unique Domains</h3>
                <h2 style="margin:0; font-size:2rem;">{}</h2>
            </div>
        """.format(df['Domain'].nunique()), unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown("""
            <div class='metric-card'>
                <h3>Avg. Response Time</h3>
                <h2 style="margin:0; font-size:2rem;">{:.2f} ms</h2>
            </div>
        """.format(df['Response Time (ms)'].mean()), unsafe_allow_html=True)

with col4:
    with st.container():
        st.markdown("""
            <div class='metric-card'>
                <h3>Success Rate</h3>
                <h2 style="margin:0; font-size:2rem;">{:.1f}%</h2>
            </div>
        """.format((df['Status Code'] < 400).mean() * 100), unsafe_allow_html=True)

# Create a mapping from IP to domain for later use
ip_map = dict(zip(df["IP Address"], df["Domain"]))
ip_list = df["IP Address"].unique()

# Geolocation of IPs
@st.cache_data(show_spinner=True)
def geolocate_ips(ip_list):
    locations = []
    for ip in ip_list:
        try:
            response = requests.get(f"http://ip-api.com/json/{ip}").json()
            if response["status"] == "success":
                locations.append({
                    "ip": ip,
                    "lat": response["lat"],
                    "lon": response["lon"],
                    "city": response["city"],
                    "country": response["country"],
                    "domain": ip_map.get(ip, "Unknown")
                })
        except:
            continue
    return pd.DataFrame(locations)

with st.spinner("Geolocating IP addresses..."):
    geo_df = geolocate_ips(ip_list)

if geo_df.empty:
    st.warning("No IPs could be geolocated.")
    st.stop()

# Main section with globe and charts
st.subheader("🌍 Global Website Distribution")

# Split the main section into two columns
col_left, col_right = st.columns([2, 1])

with col_left:
    # Plot Globe with transparent background and white outlines
    fig = go.Figure()
    
    # Add markers for each website
    fig.add_trace(go.Scattergeo(
        lon=geo_df['lon'],
        lat=geo_df['lat'],
        text=geo_df['domain'] + " (" + geo_df['ip'] + ")<br>" + geo_df['city'] + ", " + geo_df['country'],
        mode='markers',
        marker=dict(
            size=20, 
            color='#fff70a',
            opacity=1,
            symbol = 'triangle-down',
            line=dict(width=2, color='#FFFFFF')
        ),
        hoverinfo='text'
    ))
    
    # Transparent globe with white outlines
    fig.update_geos(
        projection_type="orthographic",
        showland=True,
        landcolor="rgba(8,8,8,8)",
        showocean=True,
        oceancolor="rgba(0,0,0,0)",
        showcoastlines=True,
        coastlinecolor="white",
        showcountries=False,
        countrycolor="white",
        showframe=True,
        bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(1,1,1,1)',
        height=700,
        geo=dict(
            projection_rotation=dict(lon=0, lat=0, roll=0),
            lataxis=dict(showgrid=False),
            lonaxis=dict(showgrid=False),
        )
    )
# chart_placeholder = st.empty()

# # Initial rotation angle
# rotation = 0

# while True:
#     # Update rotation angle
#     rotation = (rotation + 0.5) % 360
    
#     # Update figure
#     fig.update_layout(
#         geo=dict(projection_rotation=dict(lon=rotation))
#     )
    
#     # Re-render chart
#     chart_placeholder.plotly_chart(fig, use_container_width=True)
    
#     # Slow down rotation
#     time.sleep(0.05)


    st.plotly_chart(fig, use_container_width=True)
    st.caption("Hover to see site details. Drag to rotate the globe.")

with col_right:
    # Country distribution pie chart
    country_counts = geo_df['country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']
    
    fig_countries = px.pie(
        country_counts, 
        values='Count', 
        names='Country',
        title='Website Distribution by Country',
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    
    fig_countries.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_color='white',
        legend_font_color='white'
    )
    
    st.plotly_chart(fig_countries, use_container_width=True)

# New row for additional charts
st.subheader("📈 Performance Analytics")
col1, col2 = st.columns(2)

with col1:
    # Status code distribution
    status_counts = df['Status Code'].value_counts().reset_index()
    status_counts.columns = ['Status Code', 'Count']
    
    # Add category for status codes
    def categorize_status(code):
        if 200 <= code < 300:
            return 'Success'
        elif 300 <= code < 400:
            return 'Redirect'
        elif 400 <= code < 500:
            return 'Client Error'
        elif 500 <= code < 600:
            return 'Server Error'
        else:
            return 'Other'
    
    status_counts['Category'] = status_counts['Status Code'].apply(categorize_status)
    
    fig_status = px.bar(
        status_counts,
        x='Status Code',
        y='Count',
        color='Category',
        title='HTTP Status Code Distribution',
        color_discrete_map={
            'Success': '#4CAF50',
            'Redirect': '#2196F3',
            'Client Error': '#FF9800',
            'Server Error': '#F44336',
            'Other': '#9E9E9E'
        }
    )
    
    fig_status.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_color='white',
        xaxis_title_font_color='white',
        yaxis_title_font_color='white'
    )
    
    st.plotly_chart(fig_status, use_container_width=True)
   

with col2:
    # Response time distribution
    fig_response = px.histogram(
        df,
        x='Response Time (ms)',
        nbins=20,
        color_discrete_sequence=['#41EAD4'],
        title='Response Time Distribution (ms)',
        opacity=0.7,
        height= 300
    )
    
    
    fig_response.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_color='white',
        xaxis_title_font_color='white',
        yaxis_title_font_color='white'
    )
    
    st.plotly_chart(fig_response, use_container_width=True)
    #st.sidebar.plotly_chart(fig_response, use_container_width=True)

# Third row for additional content
col1, col2 = st.columns(2)

with col1:
    # Content length vs response time scatter plot
    fig_scatter = px.scatter(
        df,
        x='Content Length',
        y='Response Time (ms)',
        color='Depth',
        size='Content Length',
        size_max=15,
        hover_data=['Domain', 'Status Code'],
        title='Content Length vs Response Time',
        color_continuous_scale='Viridis',
        height=200
    )
    
    fig_scatter.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_color='white',
        xaxis_title_font_color='white',
        yaxis_title_font_color='white'
    )
    
    #st.plotly_chart(fig_scatter, use_container_width=True)
    st.sidebar.plotly_chart(fig_scatter, use_container_width=True)
with col2:
    # Top domains by frequency
    domain_counts = df['Domain'].value_counts().head(10).reset_index()
    domain_counts.columns = ['Domain', 'Count']
    
    fig_domains = px.bar(
        domain_counts,
        y='Domain',
        x='Count',
        orientation='h',
        title='Top 10 Most Visited Domains',
        color='Count',
        color_continuous_scale='Viridis',
        height=400
    )
    
    fig_domains.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_color='white',
        xaxis_title_font_color='white',
        yaxis_title_font_color='white',
        
    )
    
    st.plotly_chart(fig_domains, use_container_width=True)
    st.sidebar.plotly_chart(fig_domains, use_container_width=True)

# Final row with detailed data tables
st.subheader("📋 Detailed Data")
tab1, tab2 = st.tabs(["📍 IP Geolocations", "🔍 Raw Scraper Data"])

with tab1:
    st.dataframe(
        geo_df[['domain', 'ip', 'city', 'country', 'lat', 'lon']],
        use_container_width=True
    )

with tab2:
    st.dataframe(
        df,
        use_container_width=True
    )

# Footer
st.markdown("---")
st.caption(f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")