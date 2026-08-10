"""
Zepto Data & AI Platform - Module 1: Data Pipeline (/data_pipeline/pipeline.py)
Author: AI/ML Engineer (B.Tech Capstone Project)

This script performs an end-to-end Data Engineering pipeline:
1. Web Scraping: Scrapes book catalog data from books.toscrape.com across multiple categories.
2. Data Cleaning & Transformation:
   - Strips currency symbol (£) and converts price to float (price_gbp).
   - Maps text star ratings ('One'...'Five') to integer values (1-5).
   - Parses availability text into boolean integer (in_stock: 1 or 0).
   - Converts GBP to INR using fixed baseline rate: 1 GBP = 105.50 INR.
3. Database Normalization & Load:
   - Stores clean data into SQLite database (zepto_store.db) with a normalized 2-table schema.
   - Tables: 'categories' (PK: category_id) and 'books' (PK: book_id, FK: category_id).
4. SQL Queries & Pandas Equivalency Verification:
   - Executes 5 SQL queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN/IN, and JOIN.
   - Compares pd.read_sql() query output with pure in-memory pd.merge() to demonstrate equivalence.
"""

import os
import re
import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Project-defined artificial constant (Fixed Baseline Conversion Rate)
GBP_TO_INR_RATE = 105.50
BASE_URL = "http://books.toscrape.com/"
DB_PATH = os.path.join(os.path.dirname(__file__), "zepto_store.db")

RATING_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
}

def parse_rating(rating_class_list):
    """Convert text star rating (One...Five) into integer (1-5)."""
    for cls in rating_class_list:
        cls_lower = cls.lower()
        if cls_lower in RATING_MAP:
            return RATING_MAP[cls_lower]
    return 3  # Median fallback if missing/unparseable

def parse_price(price_str):
    """Strip currency symbol £ and convert price string to float."""
    try:
        clean_str = re.sub(r"[^\d.]", "", price_str)
        return float(clean_str)
    except Exception:
        return None  # Will be imputed or dropped

def parse_availability(availability_str):
    """Parse availability text into integer boolean (1 for In stock, 0 otherwise)."""
    if "in stock" in availability_str.lower():
        return 1
    return 0

def scrape_category_books(category_name, category_relative_url):
    """Scrape all books from a given category URL."""
    books = []
    cat_url = BASE_URL + category_relative_url
    
    while cat_url:
        resp = requests.get(cat_url)
        if resp.status_code != 200:
            print(f"Failed to fetch category page: {cat_url} (Status: {resp.status_code})")
            break
            
        soup = BeautifulSoup(resp.text, "html.parser")
        product_pods = soup.select("article.product_pod")
        
        for pod in product_pods:
            title_el = pod.select_one("h3 a")
            title = title_el.get("title") if title_el and title_el.get("title") else title_el.text.strip()
            
            price_el = pod.select_one("p.price_color")
            price_raw = price_el.text.strip() if price_el else ""
            
            rating_el = pod.select_one("p.star-rating")
            rating_classes = rating_el.get("class", []) if rating_el else []
            rating_raw = [c for c in rating_classes if c != "star-rating"]
            
            avail_el = pod.select_one("p.instock.availability")
            avail_raw = avail_el.text.strip() if avail_el else ""
            
            books.append({
                "title": title,
                "price_raw": price_raw,
                "rating_raw": rating_raw,
                "availability_raw": avail_raw,
                "category": category_name
            })
            
        # Pagination handling
        next_button = soup.select_one("li.next a")
        if next_button:
            next_href = next_button.get("href")
            url_parts = cat_url.split("/")
            url_parts[-1] = next_href
            cat_url = "/".join(url_parts)
        else:
            cat_url = None
            
    return books

def scrape_all_data():
    """Scrape books across at least 4 distinct categories to yield >= 60 books."""
    print("Fetching category list from books.toscrape.com...")
    resp = requests.get(BASE_URL)
    if resp.status_code != 200:
        raise RuntimeError(f"Could not access base URL: {BASE_URL}")
        
    soup = BeautifulSoup(resp.text, "html.parser")
    cat_links = soup.select("div.side_categories ul.nav-list ul li a")
    
    target_cats = ["Travel", "Mystery", "Historical Fiction", "Sequential Art", "Poetry"]
    scraped_data = []
    
    for link in cat_links:
        cat_name = link.text.strip()
        if cat_name in target_cats or len(target_cats) == 0:
            rel_url = link.get("href")
            print(f"Scraping category: '{cat_name}'...")
            cat_books = scrape_category_books(cat_name, rel_url)
            scraped_data.extend(cat_books)
            print(f" -> Scraped {len(cat_books)} books from '{cat_name}'.")
            
    print(f"Total raw books scraped: {len(scraped_data)}")
    return scraped_data

def clean_and_transform_data(raw_books):
    """Clean scraped raw text into typed DataFrames and compute price_inr."""
    df = pd.DataFrame(raw_books)
    
    # 1. Clean price
    df["price_gbp"] = df["price_raw"].apply(parse_price)
    
    # Impute missing price with median if any failed
    if df["price_gbp"].isnull().any():
        median_price = df["price_gbp"].median()
        print(f"Imputing missing price_gbp values with median: £{median_price:.2f}")
        df["price_gbp"] = df["price_gbp"].fillna(median_price)
        
    # 2. Convert price to INR using fixed rate baseline
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)
    
    # 3. Clean rating
    df["rating"] = df["rating_raw"].apply(parse_rating)
    
    # 4. Clean availability
    df["in_stock"] = df["availability_raw"].apply(parse_availability)
    
    # Select clean columns
    clean_df = df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]].copy()
    print(f"Successfully cleaned {len(clean_df)} book records.")
    return clean_df

def setup_and_load_database(clean_df, db_path=DB_PATH):
    """Create normalized 2-table SQLite schema and insert cleaned records."""
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Create Table 1: categories
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)
    
    # Create Table 2: books
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    );
    """)
    
    # Insert categories
    unique_categories = sorted(clean_df["category"].unique())
    for cat in unique_categories:
        cursor.execute("INSERT INTO categories (category_name) VALUES (?);", (cat,))
    conn.commit()
    
    # Fetch category mapping
    cursor.execute("SELECT category_name, category_id FROM categories;")
    cat_map = dict(cursor.fetchall())
    
    # Prepare books insertion
    books_to_insert = []
    for _, row in clean_df.iterrows():
        cat_id = cat_map[row["category"]]
        books_to_insert.append((
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            int(cat_id)
        ))
        
    cursor.executemany("""
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?, ?);
    """, books_to_insert)
    conn.commit()
    
    print(f"Loaded {len(unique_categories)} categories and {len(books_to_insert)} books into SQLite database '{db_path}'.")
    conn.close()

def run_sql_queries_and_verify(db_path=DB_PATH):
    """Execute required 5 SQL queries and compare SQL JOIN with pandas pd.merge()."""
    conn = sqlite3.connect(db_path)
    
    print("\n" + "="*70)
    print("EXECUTING REQUIRED SQL QUERIES AGAINST SQLITE DATABASE")
    print("="*70)
    
    # Query 1: SELECT / WHERE / ORDER BY / LIMIT
    q1 = """
    SELECT book_id, title, price_inr, rating 
    FROM books 
    WHERE price_inr > 2000.0 
    ORDER BY price_inr DESC 
    LIMIT 5;
    """
    print("\n--- QUERY 1: Top 5 Expensive Books (> 2000 INR) (SELECT/WHERE/ORDER BY/LIMIT) ---")
    df_q1 = pd.read_sql_query(q1, conn)
    print(df_q1.to_string(index=False))
    
    # Query 2: DISTINCT
    q2 = "SELECT DISTINCT rating FROM books ORDER BY rating ASC;"
    print("\n--- QUERY 2: Distinct Star Ratings in Catalog (DISTINCT) ---")
    df_q2 = pd.read_sql_query(q2, conn)
    print(df_q2.to_string(index=False))
    
    # Query 3: BETWEEN / IN
    q3 = """
    SELECT book_id, title, price_inr, rating, in_stock 
    FROM books 
    WHERE rating IN (4, 5) AND price_inr BETWEEN 1500.0 AND 4000.0 
    ORDER BY rating DESC, price_inr ASC 
    LIMIT 5;
    """
    print("\n--- QUERY 3: High-Rated Books (4-5 Stars) Priced 1500-4000 INR (IN/BETWEEN) ---")
    df_q3 = pd.read_sql_query(q3, conn)
    print(df_q3.to_string(index=False))
    
    # Query 4: JOIN (Books & Categories)
    q4 = """
    SELECT b.book_id, b.title, c.category_name, b.rating, b.price_gbp, b.price_inr
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
    WHERE b.rating = 5
    ORDER BY b.price_inr DESC
    LIMIT 10;
    """
    print("\n--- QUERY 4: Top 5-Star Books with Category Names (JOIN) ---")
    df_sql_join = pd.read_sql_query(q4, conn)
    print(df_sql_join.to_string(index=False))
    
    # Query 5: GROUP BY & Aggregation
    q5 = """
    SELECT c.category_name, COUNT(b.book_id) AS total_books, 
           ROUND(AVG(b.price_inr), 2) AS avg_price_inr, 
           MAX(b.rating) AS max_rating
    FROM categories c
    JOIN books b ON c.category_id = b.category_id
    GROUP BY c.category_name
    ORDER BY total_books DESC;
    """
    print("\n--- QUERY 5: Category Summary Statistics (GROUP BY / Aggregation) ---")
    df_q5 = pd.read_sql_query(q5, conn)
    print(df_q5.to_string(index=False))
    
    # Pandas pd.merge Equivalence Verification
    print("\n" + "="*70)
    print("PANDAS pd.merge() VS SQL JOIN EQUIVALENCE VERIFICATION")
    print("="*70)
    
    books_raw_df = pd.read_sql_query("SELECT * FROM books;", conn)
    categories_raw_df = pd.read_sql_query("SELECT * FROM categories;", conn)
    
    merged_pd = pd.merge(books_raw_df, categories_raw_df, on="category_id")
    filtered_pd = merged_pd[merged_pd["rating"] == 5].sort_values(by="price_inr", ascending=False).head(10)
    df_pandas_join = filtered_pd[["book_id", "title", "category_name", "rating", "price_gbp", "price_inr"]].reset_index(drop=True)
    
    print("\n--- Pandas pd.merge() Result ---")
    print(df_pandas_join.to_string(index=False))
    
    are_equal = df_sql_join.equals(df_pandas_join)
    print(f"\nEquivalence Check (pd.read_sql JOIN vs pd.merge): {'[SUCCESS - MATCH]' if are_equal else '[FAILED - MISMATCH]'}")
    
    conn.close()
    return are_equal

def main():
    print("Starting Module 1 Data Pipeline execution...")
    raw_data = scrape_all_data()
    clean_df = clean_and_transform_data(raw_data)
    setup_and_load_database(clean_df)
    run_sql_queries_and_verify()
    print("\nModule 1 Data Pipeline executed successfully!")

if __name__ == "__main__":
    main()
