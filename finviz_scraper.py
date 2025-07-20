import requests
from bs4 import BeautifulSoup

def fetch_finviz_reversals():
    try:
        url = "https://finviz.com/screener.ashx?v=111&f=ta_pattern_doji,ta_pattern_hammer,ta_pattern_doublebottom,ta_pattern_supptrendline"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find_all("a", class_="screener-link-primary")
        tickers = [a.text for a in table if len(a.text) <= 5]  # avoid long names
        return list(set(tickers))
    except Exception as e:
        print("❌ Failed to fetch from Finviz:", e)
        return []