import aiohttp
import asyncio
from bs4 import BeautifulSoup
import pandas as pd

# Function to fetch HTML content
async def fetch(session, url):
    try:
        async with session.get(url) as response:
            return await response.text()
    except aiohttp.ClientError as e:
        print(f"Failed to fetch URL {url}: {e}")
        return None  # You can decide to return an empty string or handle it differently

# Function to scrape individual paper details
async def scrape_paper(session, paper_url):
    html = await fetch(session, paper_url)
    if html is None:
        return None  # Early exit if html content is not fetched
    soup = BeautifulSoup(html, 'html.parser')

    # Extract title
    title = soup.find('h2').text.strip() if soup.find('h2') else "N/A"

    # Extract authors
    authors = [author.text.strip() for author in soup.find_all('li', class_='author')]

    # Extract abstract
    abstract = soup.find('p', class_='abstract').text.strip() if soup.find('p', class_='abstract') else "N/A"

    # Extract PDF link, update to 'string'
    pdf_link_element = soup.find('a', string='Download PDF')
    pdf_link = pdf_link_element['href'] if pdf_link_element else "N/A"

    return {
        'title': title,
        'authors': authors,
        'abstract': abstract,
        'pdf_link': f"https://papers.nips.cc{pdf_link}" if pdf_link != "N/A" else "N/A"
    }

# Function to scrape all papers from a specific year
async def scrape_year(session, year_url):
    html = await fetch(session, year_url)
    if html is None:
        return []  # Early exit if html content is not fetched
    soup = BeautifulSoup(html, 'html.parser')

    # Extract paper links
    paper_links = [a['href'] for a in soup.find_all('a', href=True) if '/paper/' in a['href']]

    # Scrape details for each paper
    tasks = [scrape_paper(session, f"https://papers.nips.cc{link}") for link in paper_links]
    papers = await asyncio.gather(*tasks)
    return [paper for paper in papers if paper is not None]  # Filter out None results

# Main function to scrape papers from the last 5 years
async def main():
    years = range(2018, 2023)  # Last 5 years
    base_url = "https://papers.nips.cc/paper/"

    async with aiohttp.ClientSession() as session:
        tasks = [scrape_year(session, f"{base_url}{year}") for year in years]
        results = await asyncio.gather(*tasks)

        # Flatten the list of results
        flat_results = [item for sublist in results for item in sublist if item]

        # Save results to a CSV file
        df = pd.DataFrame(flat_results)
        df.to_csv('neurips_papers.csv', index=False)
        print("Scraping completed! Data saved to 'neurips_papers.csv'.")

if __name__ == '__main__':
    asyncio.run(main())
