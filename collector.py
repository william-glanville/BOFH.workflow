import argparse
import logging
import re
import requests
import pandas as pd
import time
import common.constants as constants

from bs4 import BeautifulSoup, NavigableString
from datetime import datetime, date

rate_limiter = 0
LOG_FILE = constants.get_log_path("collection.debug.log")

logging.basicConfig(filename=LOG_FILE, filemode="w", encoding="utf-8",
                    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT = "https://www.theregister.com/"
ARCHIVE = ROOT + "Archive/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def initialize_query_table(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date)
    return pd.DataFrame({
        "Date": dates,
        "Archive": [f"{ARCHIVE}{d.strftime('%Y/%m/%d/')}" for d in dates],
        "Title": ["" for _ in dates],
        "Link": ["" for _ in dates],
        "Content": ["" for _ in dates],
    })

def prune_articles(data):
    data['Link'].replace('', None, inplace=True)
    data.dropna(subset=['Link'], inplace=True)
    return data

def populate_articles(start_date, end_date):
    data = initialize_query_table(start_date, end_date)
    total_records = len(data)
    for index, record in data.iterrows():
        time.sleep(rate_limiter)
        print(f"Processing {record.Date}...")
        constants.print_progress(index, total_records)
        page = requests.get(record.Archive, headers=HEADERS)
        if page.status_code == 200:
            soup = BeautifulSoup(page.text, "html.parser")
            articles = soup.find_all("article")
            for article in articles:
                title = article.find("h4").text.strip()
                link = article.find("a")["href"]
                if "bofh" in link:
                    data.loc[index, "Title"] = title
                    data.loc[index, "Link"] = link
    return prune_articles(data)

def extract_quoted_and_unquoted_blocks(text):
    pattern = r'(?P<quote>"[^"]*")|(?P<action>[^"]+)'
    matches = re.finditer(pattern, text)
    result = []
    for match in matches:
        if match.group("quote"):
            result.append(("quote", match.group("quote").strip()))
        else:
            result.append(("action", match.group("action").strip()))
    return result

def extract_content(link):
    print(f"Processing {ROOT + link}...")
    page = requests.get(ROOT + link, headers=HEADERS)
    if page.status_code == 200:
        soup = BeautifulSoup(page.text, "html.parser")
        body = soup.find("div", {"id": "body"})
        if body is None:
            return ""
        for tag in body.find_all():
            if tag.name != "p":
                if tag.previous_sibling and not str(tag.previous_sibling).endswith(" "):
                    tag.insert_before(NavigableString(" "))
                if tag.next_sibling and not str(tag.next_sibling).startswith(" "):
                    tag.insert_after(NavigableString(" "))
                tag.unwrap()
        paragraphs = body.find_all("p")
        content = ""
        for paragraph in paragraphs:
            text = paragraph.get_text(separator=" ", strip=True)
            blocks = extract_quoted_and_unquoted_blocks(text)
            chunk = ""
            for block in blocks:
                logging.debug(f"Block: {block}")
                chunk += block[1] + " "
            content += chunk + '\n'
        logging.debug(f"Content: {content}")
        return content
    return ""

def populate_content(data):
    for index, record in data.iterrows():
        time.sleep(rate_limiter)
        content = extract_content(record.Link)
        data.loc[index, "Content"] = content
    return data

def prune_summaries(data):
    rollup_pattern = r"(BOFH 2K: The kit and caboodle|BOFH 2000: Kit and Caboodle)"
    return data[~data["Title"].str.contains(rollup_pattern, na=False, regex=True)]

def main():
    parser = argparse.ArgumentParser(description="Fetch BOFH articles from The Register archive.")
    parser.add_argument("--start", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", type=str, help="End date in YYYY-MM-DD format")
    args = parser.parse_args()

    # Default values from original script
    default_start = date(2000, 5, 1)
    default_end = date(2025, 6, 27)

    # Choose dates: CLI > Interactive > Defaults
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        try:
            start_input = input(f"Enter start date [default: {default_start}]: ") or str(default_start)
            end_input = input(f"Enter end date [default: {default_end}]: ") or str(default_end)
            start_date = datetime.strptime(start_input, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_input, "%Y-%m-%d").date()
        except ValueError:
            print("Invalid format. Using default dates.")
            start_date, end_date = default_start, default_end

    frame = populate_articles(start_date, end_date)
    frame = populate_content(frame)
    frame = prune_summaries(frame)

    articles_path = constants.get_data_path(constants.DS_ARTICLES)
    frame.to_json(articles_path, orient="records", lines=True, force_ascii=False)
    print(f"BOFH articles saved to {articles_path}")

if __name__ == "__main__":
    main()