import requests
from bs4 import BeautifulSoup
import re
import os

# https://www.geeksforgeeks.org/downloading-pdfs-with-python-using-requests-and-beautifulsoup/
# https://medium.com/@thapliyalpiyush3/how-to-extract-website-content-using-beautifulsoup-package-in-python-b068ec1da002
# https://docs.python.org/3/library/re.html


os.makedirs("scripts", exist_ok=True)

movies = {
    "10 Things I Hate About You": "https://imsdb.com/scripts/10-Things-I-Hate-About-You.html",
    "500 Days of Summer": "https://imsdb.com/scripts/500-Days-of-Summer.html",
    "The Devil Wears Prada": "https://imsdb.com/scripts/Devil-Wears-Prada,-The.html",
    "He's Just Not That Into You": "https://imsdb.com/scripts/He's-Just-Not-That-Into-You.html",
    "The Ugly Truth": "https://imsdb.com/scripts/Ugly-Truth,-The.html",
    "Pretty Woman": "https://imsdb.com/scripts/Pretty-Woman.html",
    "A Walk to Remember": "https://imsdb.com/scripts/Walk-to-Remember,-A.html",
    "Crazy, Stupid, Love": "https://imsdb.com/scripts/Crazy,-Stupid,-Love.html",
    "Legally Blonde": "https://imsdb.com/scripts/Legally-Blonde.html",
    "Notting Hill": "https://imsdb.com/scripts/Notting-Hill.html"
}

def clean_script(script):
    script_content = re.sub(r"<.*?>", "", script)
    script_content = re.sub(r"\n+", "\n", script_content)
    script_content = re.sub(r"\s+", " ", script_content)
    script_content = script_content.strip()
    return script_content

i = 0

for title, url in movies.items():
    print(f"Downloading {title}...")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    script_content = soup.get_text()
    script_text_cleaned = clean_script(script_content)
    with open(f"scripts/{title.replace(' ', '_')}.txt", 'w', encoding='utf-8') as file:
        file.write(script_text_cleaned)
    print(f"Downloaded {title}.")
    i += 1

print(f"Downloaded {i} scripts.")
