import requests
from bs4 import BeautifulSoup
import re

# URL to scrape
url = 'https://simple.wikipedia.org/wiki/radioactivity'
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all paragraphs
paragraphs = soup.find_all('p')

# Find all headings (you can modify this if needed to include more headings like 'h2', 'h3', etc.)
headings = soup.find_all(['h1'])

# Print headings
for heading in headings:
    print(f'\n{heading.name} {heading.text}\n')

# Function to clean the text
def clean_text(text):
    # Remove all citations like [1], [2], [citation needed], etc.
    text = re.sub(r'\[\d+\]', '', text)  # Remove numbers in brackets like [1]
    text = re.sub(r'\[.*?\]', '', text)  # Remove any text in brackets (e.g., [source?], [clarification needed], etc.)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces (replaces multiple spaces with a single space)
    text = re.sub(r'\n+', ' ', text)  # Remove multiple newline characters
    text = text.strip()  # Strip leading and trailing spaces
    return text

# Apply the clean_text function to the text content of each paragraph
cleaned_paragraphs = [clean_text(paragraph.text) for paragraph in paragraphs]

# Print the cleaned text
for cleaned in cleaned_paragraphs:
    print(f'\n{cleaned}\n')
