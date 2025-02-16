import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import concurrent.futures
import time
import os
import re
import google.generativeai as genai
from typing import List, Dict
import aiohttp
import asyncio
import nest_asyncio

# Enable nested asyncio for Streamlit
nest_asyncio.apply()

# Configure session with connection pooling
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
session.mount('http://', adapter)
session.mount('https://', adapter)

async def fetch_url(url: str, session: aiohttp.ClientSession) -> str:
    """Asynchronously fetch URL content."""
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        st.error(f"Error fetching {url}: {e}")
        return ""

async def scrape_paper_details_async(paper: Dict, session: aiohttp.ClientSession) -> Dict:
    """Asynchronously scrape details for a single paper."""
    try:
        html = await fetch_url(paper['link'], session)
        if not html:
            return {'link': paper['link'], 'abstract': "N/A"}

        soup = BeautifulSoup(html, 'html.parser')
        
        abstract = "N/A"
        abstract_tag = soup.find('h4', string='Abstract')
        if abstract_tag:
            abstract_p = abstract_tag.find_next_sibling('p')
            if abstract_p:
                abstract = abstract_p.text.strip()
        
        return {'link': paper['link'], 'abstract': abstract}
    except Exception as e:
        return {'link': paper['link'], 'abstract': "N/A"}

async def scrape_papers_batch(papers: List[Dict]) -> List[Dict]:
    """Scrape details for a batch of papers concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [scrape_paper_details_async(paper, session) for paper in papers]
        results = await asyncio.gather(*tasks)
        
        for paper, details in zip(papers, results):
            paper.update(details)
        
        return papers

def scrape_neurips_page(url: str, year: int, num_papers: int) -> List[Dict]:
    """Scrape specified number of papers from a single year's page."""
    try:
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        papers = []
        paper_elements = soup.find_all('li', class_='conference')

        # Limit to specified number of papers
        for paper_element in paper_elements[:num_papers]:
            title_a = paper_element.find('a')
            if title_a:
                title = title_a.text.strip()
                link = title_a['href']
                if not link.startswith('http'):
                    link = "https://papers.nips.cc" + link

                authors_i = paper_element.find('i')
                authors = authors_i.text.strip().replace('"', '') if authors_i else "N/A"

                papers.append({
                    'title': title,
                    'link': link,
                    'authors': authors,
                    'year': year
                })

        return papers
    except Exception as e:
        st.error(f"Error scraping year {year}: {e}")
        return []

def classify_abstract(abstract: str, api_key: str) -> str:
    """Use Gemini API to classify the paper abstract."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = """
        Classify the following research paper abstract and assign the SINGLE most relevant research area label from this list:
        - Deep Learning
        - Computer Vision
        - Reinforcement Learning
        - NLP
        - Optimization

        Return only the label name, nothing else.

        Abstract:
        {abstract}
        """
        
        response = model.generate_content(prompt.format(abstract=abstract))
        return response.text.strip()
    except Exception as e:
        st.error(f"Error classifying abstract: {e}")
        return "Classification Failed"

def main():
    st.title("NeurIPS Paper Scraper and Classifier")
    
    # Initialize session states
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = None
    if 'classified_data' not in st.session_state:
        st.session_state.classified_data = None
    
    # Year selection
    st.subheader("Step 1: Select Year and Number of Papers")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_year = st.selectbox("Choose year to scrape", range(2024, 1986, -1))
    
    with col2:
        num_papers = st.number_input("Number of papers to scrape", min_value=1, max_value=10, value=3)
    
    # API key input
    st.subheader("Step 2: Enter Gemini API Key")
    api_key = st.text_input("Enter your Gemini API key", type="password")
    
    # Create columns for buttons
    col1, col2 = st.columns(2)
    
    # Scrape button
    with col1:
        if st.button("Scrape Papers"):
            with st.spinner(f"Scraping {num_papers} papers from {selected_year}..."):
                # Get year link
                base_url = "https://papers.nips.cc/"
                response = session.get(base_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                year_link = None
                
                for link in soup.find_all('a', href=lambda href: href and '/paper_files/paper/' in href):
                    if str(selected_year) in link.text:
                        year_link = "https://papers.nips.cc" + link['href']
                        break
                
                if year_link:
                    # Scrape papers list
                    papers = scrape_neurips_page(year_link, selected_year, num_papers)
                    
                    if papers:
                        # Use asyncio for concurrent scraping
                        async def process_papers():
                            return await scrape_papers_batch(papers)
                        
                        # Run async scraping
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        processed_papers = loop.run_until_complete(process_papers())
                        
                        # Convert to DataFrame
                        st.session_state.scraped_data = pd.DataFrame(processed_papers)
                        st.success(f"Scraped {len(processed_papers)} papers from {selected_year}")
                        
                        # Show download button
                        csv = st.session_state.scraped_data.to_csv(index=False)
                        st.download_button(
                            label="Download Scraped Data",
                            data=csv,
                            file_name=f"neurips_{selected_year}_scraped.csv",
                            mime="text/csv"
                        )
    
    # Annotate button
    with col2:
        if st.button("Annotate Papers", disabled=not (st.session_state.scraped_data is not None and api_key)):
            if st.session_state.scraped_data is not None:
                with st.spinner("Classifying papers..."):
                    # Create a copy of the data with only title and the new label column
                    classified_data = st.session_state.scraped_data[['title']].copy()
                    
                    # Classify each abstract
                    labels = []
                    for abstract in st.session_state.scraped_data['abstract']:
                        label = classify_abstract(abstract, api_key)
                        labels.append(label)
                    
                    classified_data['label'] = labels
                    st.session_state.classified_data = classified_data
                    
                    # Show download button for classified data
                    csv = classified_data.to_csv(index=False)
                    st.download_button(
                        label="Download Classified Data",
                        data=csv,
                        file_name=f"neurips_{selected_year}_classified.csv",
                        mime="text/csv"
                    )
    
    # Display data tables
    if st.session_state.scraped_data is not None:
        st.subheader("Scraped Papers")
        st.dataframe(st.session_state.scraped_data)
    
    if st.session_state.classified_data is not None:
        st.subheader("Classified Papers")
        st.dataframe(st.session_state.classified_data)
    
    # Instructions
    st.markdown("""
    ### How to use:
    1. Select the year and number of papers you want to analyze
    2. Enter your Gemini API key
    3. Click 'Scrape Papers' to collect the papers
    4. Click 'Annotate Papers' to classify them
    5. Download the results using the download buttons
    
    ### How to get a Gemini API key:
    1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. Sign in with your Google account
    3. Click "Create API Key"
    4. Copy the generated key and paste it above
    """)

if __name__ == "__main__":
    main()
