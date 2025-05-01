import requests
from bs4 import BeautifulSoup
 
url= 'https://simple.wikipedia.org/wiki/radioactivity'
response=requests.get(url)

soup=BeautifulSoup(response.text,'html.parser')
paragraphs=soup.find_all('p')

headings = soup.find_all(['h1','h2','h3','h4','h5','h6'])
for heading in headings:
    print(f'\n{heading.name} {heading.text}\n')
for paragraph in paragraphs:
    print(f'\n {paragraph.text}\n')