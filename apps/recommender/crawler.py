import os
import time
import requests
import logging
from bs4 import BeautifulSoup
from neo4j import GraphDatabase
from typing import List, Dict, Optional
import json
import re
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "oburus_pass")

# Optional: LLM Configuration
LLM_API_KEY = os.getenv("LLM_API_KEY")

class MenuCrawler:
    def __init__(self, driver):
        self.driver = driver
        self.session = self.driver.session()

    def close(self):
        self.session.close()

    def construct_tripadvisor_url(self, restaurant_id, name):
        """
        Constructs a likely valid TripAdvisor URL for Istanbul restaurants.
        Schema: https://www.tripadvisor.com.tr/Restaurant_Review-g293974-d{ID}-Reviews-{NAME}-Istanbul.html
        """
        # Clean name: Replace spaces with underscores, remove special chars
        safe_name = re.sub(r'[^\w\s]', '', name).strip().replace(" ", "_")
        return f"https://www.tripadvisor.com.tr/Restaurant_Review-g293974-d{restaurant_id}-Reviews-{safe_name}-Istanbul.html"

    def get_restaurants_to_crawl(self, limit: int = 10):
        """
        Fetches restaurants that haven't been crawled recently or at all.
        """
        # Patch for known URLs missing in the DB
        KNOWN_URLS = {
            "Dervis Cafe & Restaurant": "http://derviscafe2.com",
            "Akın Restoran": "https://akinrestoran.dijital.menu/",
        }

        query = """
        MATCH (r:Restaurant)
        WHERE r.last_crawled IS NULL OR r.last_crawled < datetime() - duration('P7D')
        RETURN r.id AS id, r.name AS name, r.website AS website
        ORDER BY r.name IN ['Dervis Cafe & Restaurant', 'Akın Restoran'] DESC
        LIMIT $limit
        """
        result = self.session.run(query, limit=limit)
        
        restaurants = []
        for record in result:
            r = {"id": record["id"], "name": record["name"], "website": record["website"]}
            
            # 1. Try Patch
            if not r["website"] and r["name"] in KNOWN_URLS:
                r["website"] = KNOWN_URLS[r["name"]]
                logger.info(f"Using known URL for {r['name']}: {r['website']}")
            
            restaurants.append(r)
            
        return restaurants

    def fetch_page_content(self, url: str) -> Optional[str]:
        """
        Fetches the raw HTML content of a page.
        """
        if not url:
            return None
        
        try:
            # Enhanced headers to mimic a real Chrome browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.tripadvisor.com.tr/',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            # Force encoding if needed, or let requests handle it
            response.encoding = response.apparent_encoding
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def parse_menu_basic(self, html_content: str) -> List[Dict]:
        """
        Basic extraction using BeautifulSoup. 
        """
        soup = BeautifulSoup(html_content, 'lxml')
        items = []
        
        # DEBUG: Log structure
        full_text = soup.get_text(" ", strip=True)
        logger.info(f"DEBUG: Content Preview (200 chars): {full_text[:200]}")

        BLACKLIST_PHRASES = [
            "rezervasyon", "sosyal medya", "iletişim", "contact", "about us", "hakkımızda",
            "info@", ".com", "tel:", "adres", "address", "follow us", "takip et", "hoşgeldiniz",
            "welcome", "menu", "menü", "home", "anasayfa", "gallery", "galeri", "english", "türkçe",
            "dilerseniz", "copyright", "rights reserved"
        ]

        # --- STRATEGY 1: Container-based Parsing (New) ---
        # Look for containers that likely hold a single menu item
        candidate_containers = soup.find_all(['div', 'li', 'tr'], class_=re.compile(r'(menu|food|dish|product|item)', re.IGNORECASE))
        
        for container in candidate_containers:
            # Get all text in this container
            c_text = container.get_text(" ", strip=True)
            if len(c_text) < 5 or len(c_text) > 500: # Sanity check length
                continue
            
            # Check blacklist on the whole container text
            if any(phrase in c_text.lower() for phrase in BLACKLIST_PHRASES):
                continue

            # Attempt to find Price
            price = 0
            price_match = re.search(r'(\d{1,5})\s*(?:₺|TL)', c_text)
            if price_match:
                price = int(price_match.group(1))

            # Attempt to find Name
            # Heuristic: The first header tag (h1-h6) or the first bold tag, or just the start of the text
            name = ""
            header = container.find(['h3', 'h4', 'h5', 'h6', 'strong', 'b'])
            if header:
                name = header.get_text(" ", strip=True)
            else:
                # Split by newlines or predictable separators if no header
                parts = c_text.splitlines()
                if parts:
                    name = parts[0].strip()
            
            # Cleanup Name
            name = re.sub(r'\d{1,5}\s*(?:₺|TL)', '', name).strip(" .-:•")
            
            # If we found a decent name and (price OR it looks like a list item)
            if len(name) > 2 and len(name) < 100:
                # Try to find ingredients/description (everything else)
                desc = c_text.replace(name, "").replace(price_match.group(0) if price_match else "", "").strip(" .-:•()")
                ingredients = [i.strip() for i in desc.split(',') if len(i.strip()) > 2]
                
                # Filter out garbage names that might have slipped through
                if not any(x in name.lower() for x in ["fiyat", "tl", "price"]):
                    items.append({
                        "dish": name,
                        "price": price,
                        "ingredients": ingredients
                    })
                    logger.info(f"Scraped Dish (Container): {name} | Price: {price}")

        # --- STRATEGY 2: Line-by-Line / Tag-by-Tag Parsing (Original) ---
        # Scan all text blocks
        for element in soup.find_all(['p', 'div', 'li', 'span', 'h3', 'h4', 'td']):
            text = element.get_text(" ", strip=True)
            
            if len(text) < 5:
                continue

            # Check blacklist
            text_lower = text.lower()
            if any(phrase in text_lower for phrase in BLACKLIST_PHRASES):
                continue

            # PATTERN 1: Dervis Cafe Style (Concatenated)
            if "...₺" in text:
                try:
                    # Simplified Regex: Name (lazy), dots, optional Currency, Ingredients
                    # Capture everything before dots as name, everything after as ingredients
                    pattern = r'(.+?)\.{2,}[₺TL]?\s*(.+)'
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        dish_name = match.group(1).strip()
                        ingredients_raw = match.group(2)
                        ingredients = [i.strip() for i in ingredients_raw.split(',') if i.strip()]
                        
                        if len(dish_name) > 2 and len(ingredients) > 0:
                            items.append({
                                "dish": dish_name,
                                "price": 0,
                                "ingredients": ingredients
                            })
                            logger.info(f"Scraped Dish (Type 1): {dish_name} | Ingredients: {ingredients}")
                    continue
                except Exception as e:
                    logger.error(f"Regex Type 1 failed on '{text[:50]}...': {e}")

            # PATTERN 2: Standard Name (Ingredients)
            try:
                # Regex: Name (not ( or digit), dots/space, (Ingredients)
                # Escaping parentheses carefully: \( and \)
                pattern = r'([^(\d]+?)(?:\.{2,}|\s{2,}|₺|TL|\d).*?\(([^)]+)\)'
                matches = re.finditer(pattern, text)
                found_any = False
                for match in matches:
                    found_any = True
                    dish_name = match.group(1).strip(" .₺-*•:\t\n")
                    ingredients_text = match.group(2)
                    ingredients = [i.strip() for i in ingredients_text.split(',')]
                    
                    if len(dish_name) > 2 and len(ingredients) >= 1:
                        items.append({"dish": dish_name, "price": 0, "ingredients": ingredients})
                        logger.info(f"Scraped Dish (Type 2): {dish_name}")
                
                if found_any:
                    continue
            except Exception as e:
                logger.error(f"Regex Type 2 failed on '{text[:50]}...': {e}")

            # PATTERN 3: Simple Name (Ingredients)
            try:
                pattern = r'^([^(\d]+?)\s+\(([^)]+)\)$'
                match = re.search(pattern, text)
                if match:
                    dish_name = match.group(1).strip(" .₺-*•:\t\n")
                    ingredients_text = match.group(2)
                    ingredients = [i.strip() for i in ingredients_text.split(',')]
                    items.append({"dish": dish_name, "price": 0, "ingredients": ingredients})
                    logger.info(f"Scraped Dish (Type 3): {dish_name}")
            except Exception as e:
                 logger.error(f"Regex Type 3 failed on '{text[:50]}...': {e}")

            # PATTERN 4: Digital Menu Simple (Name PriceCurrency)
            # e.g. "Serpme Kahvaltı 450 TL"
            try:
                # Look for Name followed by Price at the end of the string
                pattern = r'^(.+?)\s+(\d{1,5})\s*(?:₺|TL)$'
                match = re.search(pattern, text)
                if match:
                    dish_name = match.group(1).strip(" .₺-*•:\t\n")
                    # Avoid capturing generic labels like "Fiyat" or descriptions that are too long
                    if 3 < len(dish_name) < 60 and not any(x in dish_name.lower() for x in ["fiyat", "price", "toplam"]):
                        price = int(match.group(2))
                        items.append({"dish": dish_name, "price": price, "ingredients": []})
                        logger.info(f"Scraped Dish (Type 4): {dish_name} | Price: {price}")
            except Exception as e:
                logger.error(f"Regex Type 4 failed on '{text[:50]}...': {e}")

        # Remove duplicates
        unique_items = {}
        for item in items:
            if item["dish"] not in unique_items:
                unique_items[item["dish"]] = item
        
        return list(unique_items.values())

    def parse_menu_with_llm(self, html_content: str) -> List[Dict]:
        """
        Advanced extraction using an LLM.
        """
        if not LLM_API_KEY:
            logger.warning("LLM_API_KEY not set. Skipping LLM extraction.")
            return []

        prompt = f"""
        Extract menu items from the following HTML. 
        Return JSON list: [{{ "dish": "name", "price": number, "ingredients": ["ing1", "ing2"] }}]
        HTML: {html_content[:2000]}...
        """
        
        logger.info("Sending request to LLM...")
        return []

    def crawl_dijital_menu(self, main_html: str, base_url: str) -> List[Dict]:
        """
        Specialized crawler for dijital.menu sites that use widgets/categories.
        """
        soup = BeautifulSoup(main_html, 'lxml')
        all_items = []
        
        # Find category links (widgets)
        # Look for <a> tags with specific class or just general structure if classes change
        # User reported: <a class="menu-grid-item style-4" ...>
        category_links = set()
        for a in soup.find_all('a', class_=lambda c: c and 'menu-grid-item' in c):
            href = a.get('href')
            if href:
                category_links.add(urljoin(base_url, href))
        
        if not category_links:
            logger.info("No dijital.menu categories found on main page. Trying basic parse on main page.")
            return self.parse_menu_basic(main_html)

        logger.info(f"Found {len(category_links)} categories in Dijital Menu.")
        
        for link in category_links:
            logger.info(f"Fetching category: {link}")
            cat_html = self.fetch_page_content(link)
            if cat_html:
                # Reuse basic parser for the category page
                items = self.parse_menu_basic(cat_html)
                if items:
                    logger.info(f"Found {len(items)} items in category.")
                    all_items.extend(items)
                else:
                    logger.warning(f"No items found in category {link}")
                
                time.sleep(0.5) # Be polite
                
        # Remove duplicates based on dish name
        unique_items = {}
        for item in all_items:
            if item["dish"] not in unique_items:
                unique_items[item["dish"]] = item
                
        return list(unique_items.values())

    def crawl_generic_menu(self, main_html: str, base_url: str) -> List[Dict]:
        """
        Generic crawler that looks for 'Menu' or 'Menü' links on the main page
        and follows them to extract items.
        """
        soup = BeautifulSoup(main_html, 'lxml')
        menu_links = set()
        
        # Search 1: <a> tags with text containing 'menu' or 'menü'
        for a in soup.find_all('a', href=True):
            text = a.get_text(" ", strip=True).lower()
            href = a.get('href')
            if "menu" in text or "menü" in text or "menu" in href.lower() or "menü" in href.lower():
                full_url = urljoin(base_url, href)
                # Avoid self-loops or linking back to home
                if full_url != base_url and len(full_url) > len(base_url):
                    menu_links.add(full_url)
        
        if not menu_links:
            return []

        logger.info(f"Found potential menu links: {menu_links}")
        
        all_items = []
        for link in menu_links:
            logger.info(f"Following menu link: {link}")
            html = self.fetch_page_content(link)
            if html:
                items = self.parse_menu_basic(html)
                if items:
                    all_items.extend(items)
                time.sleep(0.5)

        # Remove duplicates
        unique_items = {}
        for item in all_items:
            if item["dish"] not in unique_items:
                unique_items[item["dish"]] = item
                
        return list(unique_items.values())

    def update_graph(self, restaurant_id: int, menu_items: List[Dict]):
        """
        Updates the Neo4j graph with:
        (Restaurant)-[:SERVES]->(Dish)-[:HAS_INGREDIENT]->(Ingredient)
        """
        if not menu_items:
            logger.info(f"No items to update for restaurant {restaurant_id}.")
            return

        query = """
        MATCH (r:Restaurant {id: $restaurant_id})
        SET r.last_crawled = datetime()
        
        WITH r
        UNWIND $menu_items AS item
        MERGE (d:Dish {name: item.dish, restaurant_id: $restaurant_id})
        SET d.price = item.price
        MERGE (r)-[:SERVES]->(d)
        
        WITH d, item
        UNWIND item.ingredients AS ing_name
        MERGE (i:Ingredient {name: ing_name})
        MERGE (d)-[:HAS_INGREDIENT]->(i)
        """
        
        self.session.run(query, restaurant_id=restaurant_id, menu_items=menu_items)
        logger.info(f"Updated graph for restaurant {restaurant_id} with {len(menu_items)} items.")

    def run(self):
        logger.info("Starting Offline Crawler...")
        
        # 1. Get Candidates
        restaurants = self.get_restaurants_to_crawl()
        logger.info(f"Found {len(restaurants)} restaurants to crawl.")
        
        for r in restaurants:
            logger.info(f"Processing {r['name']}...")
            
            menu_data = []
            
            # 2. Fetch & Extract
            if r['website']:
                html = self.fetch_page_content(r['website'])
                if html:
                    # Special handling for dijital.menu
                    if "dijital.menu" in r['website']:
                        menu_data = self.crawl_dijital_menu(html, r['website'])
                    # Try LLM first if available, else basic
                    elif LLM_API_KEY:
                        menu_data = self.parse_menu_with_llm(html)
                    else:
                        menu_data = self.parse_menu_basic(html)
                    
                    # Fallback: If basic parse found nothing, try looking for Menu links
                    if not menu_data:
                        logger.info("Basic parse yielded no items. Trying generic menu discovery...")
                        menu_data = self.crawl_generic_menu(html, r['website'])

                else:
                    logger.warning(f"Could not fetch content for {r['name']}")
            else:
                logger.info(f"No website for {r['name']}, skipping.")
            
            # 3. Update Graph
            if menu_data:
                self.update_graph(r['id'], menu_data)
            else:
                self.session.run("MATCH (r:Restaurant {id: $id}) SET r.last_crawled = datetime()", id=r['id'])
                logger.info(f"No menu data found for {r['name']}, marked as crawled.")
            
            # Sleep to be polite
            time.sleep(1)

        logger.info("Crawl cycle finished.")

if __name__ == "__main__":
    # Wait for Neo4j to be ready (optional, for docker-compose scenarios)
    time.sleep(2) 
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    crawler = MenuCrawler(driver)
    
    try:
        crawler.run()
    except Exception as e:
        logger.error(f"Crawler failed: {e}")
    finally:
        crawler.close()
        driver.close()
