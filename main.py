import time
import requests
import pandas as pd
from tqdm import tqdm
from bs4.element import Tag
from bs4 import BeautifulSoup

BASE_URL_GRADUATES = "https://graduation.apps.binus.ac.id/edition/wisuda-73/graduates/"
BASE_URL_OUTSTANDING = "https://graduation.apps.binus.ac.id/edition/wisuda-70/outstanding-graduates/"
params = {
    "SearchGraduate": "",
    "GraduatesDegree": 3, # Strata 1
    "GraduatesFaculty": 6, # School of Computer Science
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36"
}

def fetch_soup(url: str, params=None) -> BeautifulSoup:
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def extract_thesis_title(url:str) -> str:
    soup = fetch_soup(url)
    graduates_meta = soup.select("div.graduates-meta")

    return graduates_meta[-1].select_one(".table-cell .the-value").get_text()

def scrape_list_page(page: Tag) -> list:    
    results = []
    for li in tqdm(page.select("li.the-student")):
        name = li.select_one("span.student-name").get_text(strip=True)
        ipk = li.select_one(".gpa-row .col-xs-6 .value").get_text(strip=True)
        program = li.select_one("span.program-name").get_text(strip=True)
        profile_url = li.select_one("a.photo")["href"]
        thesis_title = extract_thesis_title(profile_url)

        graduate = {
            "Name": name,
            # "IPK": ipk,
            "Program": program,
            "Thesis Title" : thesis_title
        }

        results.append(graduate)
    
    return results

def main():
    page = 1
    df = pd.DataFrame()
    
    while True:
        print(f"Scraping page {page}..")

        params["page"] = page 
        soup = fetch_soup(BASE_URL_OUTSTANDING, params)

        graduates_list = soup.find("ul", id="the-graduates")
        if not graduates_list:
            print("No more pages. Stopping..")
            break
        
        graduates = scrape_list_page(graduates_list)
        df = pd.concat([df, pd.DataFrame(graduates)], ignore_index=True)
        page += 1 

    df.to_csv("70-Graduates.csv", index=False, header=True)
    print(f"Finishied Scraping in total scrapped data {len(df)}")

if __name__ == "__main__":
    main()