import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import aiohttp
import asyncio
import nest_asyncio
import google.generativeai as genai

# Enable nested asyncio for Streamlit
nest_asyncio.apply()

# Set up a requests session with connection pooling
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
session.mount('http://', adapter)
session.mount('https://', adapter)

async def fetch_url(url: str, session: aiohttp.ClientSession, retries: int = 3) -> str:
    """Asynchronously fetch URL content with a custom User-Agent, retry logic, and timeout."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0 Safari/537.36"
    }
    for attempt in range(retries):
        try:
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    st.error(f"Error: Received status code {response.status} for {url}")
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(1)  # Wait before retrying
            else:
                st.error(f"Error fetching {url}: {e}")
                return ""

async def scrape_paper_details_async(paper: dict, session: aiohttp.ClientSession) -> dict:
    """Asynchronously scrape details (abstract) for a single paper."""
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

async def scrape_papers_batch(papers: list) -> list:
    """Scrape details for a batch of papers concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [scrape_paper_details_async(paper, session) for paper in papers]
        results = await asyncio.gather(*tasks)
        for paper, details in zip(papers, results):
            paper.update(details)
        return papers

def scrape_neurips_page(url: str, year: int, num_papers: int = None) -> list:
    """
    Scrape papers from a given year's NeurIPS page.
    If num_papers is None, all papers on the page are returned.
    """
    try:
        response = session.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        papers = []
        paper_elements = soup.find_all('li', class_='conference')
        if num_papers is not None:
            paper_elements = paper_elements[:num_papers]
        for paper_element in paper_elements:
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
    st.title("NeurIPS Papers Scraper and Classifier")
    
    if st.button("Download papers data"):
        # Years to scrape all papers from
        years = [2022, 2023, 2024]
        all_papers = []
        base_url = "https://papers.nips.cc/"
        
        with st.spinner("Scraping papers from NeurIPS..."):
            try:
                response = session.get(base_url)
                response.raise_for_status()
                main_soup = BeautifulSoup(response.content, 'html.parser')
            except Exception as e:
                st.error(f"Error accessing NeurIPS main page: {e}")
                return
            
            # For each year, locate the corresponding link and scrape all papers.
            for year in years:
                year_link = None
                for link in main_soup.find_all('a', href=True):
                    if str(year) in link.text:
                        href = link['href']
                        if not href.startswith("http"):
                            href = base_url.rstrip("/") + href
                        year_link = href
                        break
                if not year_link:
                    st.error(f"Could not find link for year {year}.")
                    continue
                papers = scrape_neurips_page(year_link, year, num_papers=None)
                if papers:
                    all_papers.extend(papers)
                else:
                    st.error(f"No papers found for year {year}.")
        
        if not all_papers:
            st.error("No papers were scraped.")
            return
        
        # Asynchronously fetch paper details (e.g., abstracts)
        async def process_all_papers():
            return await scrape_papers_batch(all_papers)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        processed_papers = loop.run_until_complete(process_all_papers())
        df_scraped = pd.DataFrame(processed_papers)
        
        # Replace with your actual Gemini API key
        API_KEY = "AIzaSyBPuNUYcXYoz7Rwz_0pvuSFrD4_sC5R_5g"
        with st.spinner("Classifying paper abstracts..."):
            labels = []
            for abstract in df_scraped['abstract']:
                label = classify_abstract(abstract, API_KEY)
                labels.append(label)
            df_scraped['label'] = labels
        
        st.subheader("Annotated Papers")
        st.dataframe(df_scraped)
        
        csv_data = df_scraped.to_csv(index=False)
        st.download_button(
            label="Download Annotated Data",
            data=csv_data,
            file_name="neurips_2022_2023_2024_annotated.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
