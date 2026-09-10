"""
Unit tests for Module 1 Data Pipeline (/data_pipeline/test_pipeline.py)
Author: Kammari Hemanth Kumar Achari
"""

import os
import sys
import sqlite3
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from pipeline import (
    parse_rating,
    parse_price,
    parse_availability,
    GBP_TO_INR_RATE,
    DB_PATH,
    scrape_all_data,
    clean_and_transform_data,
    setup_and_load_database
)

class TestDataPipeline(unittest.TestCase):
    
    def test_parse_rating(self):
        self.assertEqual(parse_rating(["star-rating", "Three"]), 3)
        self.assertEqual(parse_rating(["star-rating", "One"]), 1)
        self.assertEqual(parse_rating(["star-rating", "Five"]), 5)
        self.assertEqual(parse_rating(["unknown-class"]), 3)  # Fallback median rating
        
    def test_parse_price(self):
        self.assertEqual(parse_price("£51.77"), 51.77)
        self.assertEqual(parse_price("£10.00"), 10.00)
        self.assertIsNone(parse_price("invalid_price"))
        
    def test_parse_availability(self):
        self.assertEqual(parse_availability("In stock (22 available)"), 1)
        self.assertEqual(parse_availability("Out of stock"), 0)
        
    def test_gbp_to_inr_conversion(self):
        price_gbp = 10.0
        expected_inr = round(10.0 * 105.50, 2)
        self.assertEqual(expected_inr, 1055.0)
        self.assertEqual(GBP_TO_INR_RATE, 105.50)
        
    def test_end_to_end_pipeline_and_db_schema(self):
        # Scrape raw data
        raw_data = scrape_all_data()
        self.assertGreaterEqual(len(raw_data), 60, "Scraped raw books count should be at least 60.")
        
        # Clean data
        clean_df = clean_and_transform_data(raw_data)
        self.assertIn("price_inr", clean_df.columns)
        self.assertTrue(clean_df["rating"].between(1, 5).all(), "Ratings must be between 1 and 5.")
        self.assertTrue(clean_df["in_stock"].isin([0, 1]).all(), "Availability must be 0 or 1.")
        
        # Load database
        setup_and_load_database(clean_df, DB_PATH)
        self.assertTrue(os.path.exists(DB_PATH), "Database file zepto_store.db must exist.")
        
        # Test Database schema & PK/FK constraints
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM categories;")
        cat_count = cursor.fetchone()[0]
        self.assertGreaterEqual(cat_count, 3, "Categories count should be >= 3.")
        
        cursor.execute("SELECT COUNT(*) FROM books;")
        book_count = cursor.fetchone()[0]
        self.assertGreaterEqual(book_count, 60, "Books count should be >= 60.")
        
        # Check FK integrity
        cursor.execute("""
            SELECT b.book_id, c.category_name 
            FROM books b 
            JOIN categories c ON b.category_id = c.category_id;
        """)
        join_rows = cursor.fetchall()
        self.assertEqual(len(join_rows), book_count, "All books must successfully join with their parent category.")
        
        conn.close()

if __name__ == "__main__":
    unittest.main()
