# Module 1: Data Engineering Pipeline (`/data_pipeline`)

**Module Marks:** 25 Marks  
**Author:** AI/ML Engineer (B.Tech Capstone Project)  
**Target Domain:** Catalog & Competitive Pricing Pipeline  

---

## 📌 Executive Summary
This module implements an automated **Data Engineering Pipeline** that scrapes live catalog pricing data from [books.toscrape.com](http://books.toscrape.com/), cleans and standardizes the fields, converts prices using a fixed baseline exchange rate, loads the structured dataset into a normalized **SQLite relational database**, and verifies data integrity using both **SQL queries** and **Pandas DataFrames**.

---

## ⚙️ Baseline Currency Conversion Rate
* **Required Fixed Rate:** `1 GBP = 105.50 INR`
* **Note:** This is an artificial, project-defined constant for this capstone assignment (requiring no external network API or date reference). All prices in INR (`price_inr`) are calculated strictly as `price_gbp * 105.50`.

---

## 🏗️ Pipeline Architecture & Design Decisions

### 1. Web Scraping (`requests` + `BeautifulSoup`)
* Scrapes catalog listings across **5 distinct categories**: *Travel*, *Mystery*, *Historical Fiction*, *Sequential Art*, and *Poetry*.
* Captures 5 raw attributes per book: `title`, `price_raw`, `rating_raw`, `availability_raw`, and `category`.
* Handles web pagination seamlessly by following `<li class="next"><a href="...">` links until all category pages are retrieved ($\ge 60$ total books).

### 2. Data Cleaning & Type Transformation
* **Price Parsing (`price_gbp`):** Strips the currency symbol (`£`) using regular expressions `re.sub(r"[^\d.]", "", price_str)` and converts to floating-point numbers.
* **Missing Value Imputation:** If any row fails numeric parsing, median imputation (`df['price_gbp'].median()`) is applied to prevent pipeline crashes.
* **Rating Mapping (`rating`):** Maps string ratings (`"One"..."Five"`) to integer values (`1...5`). If missing/unparseable, defaults to median rating (`3`).
* **Stock Parsing (`in_stock`):** Converts string availability (`"In stock (22 available)"`) into binary integer boolean (`1` for in stock, `0` otherwise).
* **Currency Conversion (`price_inr`):** Multiplies `price_gbp` by `105.50` and rounds to 2 decimal places.

---

## 🗄️ Database Schema (Normalized 2-Table PK / FK)

The SQLite database (`zepto_store.db`) enforces a normalized relational structure with Foreign Key constraints enabled (`PRAGMA foreign_keys = ON;`).

```sql
-- Table 1: Categories (Parent Table)
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

-- Table 2: Books (Child Table referencing Categories)
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
```

---

## 📊 Executed SQL Queries & Summary

The script executes 5 distinct SQL queries against SQLite:
1. **Query 1 (SELECT/WHERE/ORDER BY/LIMIT):** Top 5 most expensive books in INR (`WHERE price_inr > 2000.0 ORDER BY price_inr DESC LIMIT 5`).
2. **Query 2 (DISTINCT):** Distinct star ratings present in catalog (`SELECT DISTINCT rating FROM books`).
3. **Query 3 (IN/BETWEEN):** High-rated books (4-5 stars) priced between 1500 and 4000 INR (`WHERE rating IN (4, 5) AND price_inr BETWEEN 1500.0 AND 4000.0`).
4. **Query 4 (JOIN):** Inner join between `books` and `categories` listing top 10 5-star books with category names.
5. **Query 5 (GROUP BY / Aggregation):** Total books count and average price in INR per category (`GROUP BY c.category_name`).

### Pandas `pd.read_sql` vs `pd.merge` Equivalence
Query 4 output loaded via `pd.read_sql()` was compared against pure in-memory `pd.merge(books_df, categories_df, on='category_id')`. Both approaches yielded identical data structures, validating database load accuracy.

---

## 🚀 Execution Instructions

### Run Pipeline End-to-End:
```bash
python data_pipeline/pipeline.py
```

### Run Unit Tests:
```bash
python -m unittest data_pipeline/test_pipeline.py
```
