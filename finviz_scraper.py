
import requests
from bs4 import BeautifulSoup

def fetch_finviz_reversals():
    url = "https://finviz.com/screener.ashx?v=111&f=ta_pattern_doubbottom,ta_pattern_doji,ta_pattern_hammer,ta_pattern_trendline"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.select("table.table-dark-row-cp tr[valign='top']")
        tickers = [row.select_one("a.screener-link-primary").text.strip() for row in rows if row.select_one("a.screener-link-primary")]
        return tickers
    except Exception as e:
        print(f"❌ Failed to fetch Finviz data: {e}")
        return []
