import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

url = "https://www.arbeitsagentur.de/ueber-uns/veroeffentlichungen/merkblaetter-downloads"

#If there is no such folder, the script will create one automatically
# folder_location = r'webscraping_afa'
# if not os.path.exists(folder_location):
#     os.mkdir(folder_location)

response = requests.get(url)

if response.status_code == 200:
    html_content = response.text
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    exit()

soup = BeautifulSoup(response.text, "html.parser")

i = 1
tot_files = len(soup.select("a[href$='.pdf']"))

for link in soup.select("a[href$='.pdf']"):
    # Name the pdf files using the last portion of each link which are unique in this case
    # filename = os.path.join(folder_location,link['href'].split('/')[-1])
    filename = os.path.join(link['href'].split('/')[-1])
    print(filename)
    print(f"Downloading file {i} of {tot_files}")
    with open(filename, 'wb') as f:
        f.write(requests.get(urljoin(url,link['href'])).content)
    i += 1
