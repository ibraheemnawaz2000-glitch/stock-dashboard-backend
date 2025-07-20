
import requests
from bs4 import BeautifulSoup

def fetch_finviz_reversals():
    patterns = {
        "Hammer": "ta_pattern_hammer",
        "Doji": "ta_pattern_doji",
        "Double Bottom": "ta_pattern_doublebottom",
        "Trendline Support": "ta_pattern_tl_support"
    }

    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://finviz.com/screener.ashx?v=111&f={}"

    tickers = set()

    for name, pattern in patterns.items():
        url = base_url.format(pattern)
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.select("a.screener-link-primary")
            for link in links:
                ticker = link.text.strip().upper()
                if 1 <= len(ticker) <= 5:
                    tickers.add(ticker)
        except Exception as e:
            print(f"Error fetching {name} pattern: {e}")

    return list(tickers)
