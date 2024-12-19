from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
import json
import time
from datetime import datetime, timedelta
from openai import OpenAI
import httpx
from typing import Dict, Set, Optional, List
from dataclasses import dataclass
import os
import hashlib
from .utils import setup_logging

@dataclass
class PageContent:
    url: str
    title: str
    content: str
    links: List[dict]

class Explorer:
    def __init__(self, school: str, start_url: str):
        self.school = school
        self.start_url = start_url
        self.base_domain = urlparse(start_url).netloc
        self.visited = set()
        self.page_importance = {}
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            http_client=httpx.Client()
        )
        
        # Rate limiting
        self.total_tokens = 0
        self.minute_tokens = 0
        self.last_request_time = datetime.now()
        self.TPM_LIMIT = 30000
        self.MAX_PAGES = 500
        
        self.logger = setup_logging(school, "explorer")
        os.makedirs(f"{school}/pages", exist_ok=True)
        
    def fetch_page(self, url: str) -> Optional[PageContent]:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
                
            links = []
            for a in soup.find_all('a', href=True):
                href = urljoin(url, a['href'])
                if self.base_domain in href and '#' not in href:
                    links.append({
                        'url': href,
                        'text': a.get_text().strip()
                    })
                    
            return PageContent(
                url=url,
                title=soup.title.string if soup.title else "",
                content=soup.get_text(separator=' ', strip=True),
                links=links
            )
            
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def check_rate_limit(self) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        if self.last_request_time < minute_ago:
            self.minute_tokens = 0
            
        return self.minute_tokens < self.TPM_LIMIT

    def analyze_page(self, content: PageContent) -> dict:
        if not self.check_rate_limit():
            time.sleep(60)
        
        try:
            messages = [
                {"role": "system", "content": """Analyze medical school webpage content.
                Return JSON with:
                {
                    "importance_score": 0-1,
                    "explorer_tags": ["admissions", "requirements", etc],
                    "abstract": "100-word summary of key content",
                    "recommended_links": [{"url": "url", "priority": "high/medium/low"}]
                }"""},
                {"role": "user", "content": json.dumps({
                    "url": content.url,
                    "title": content.title,
                    "content": content.content[:2000]
                })}
            ]
            
            self.last_request_time = datetime.now()
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1
            )
            
            tokens = response.usage.total_tokens
            self.total_tokens += tokens
            self.minute_tokens += tokens
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            self.logger.error(f"Error analyzing {content.url}: {str(e)}")
            return {
                "importance_score": 0,
                "explorer_tags": [],
                "abstract": "",
                "recommended_links": []
            }

    def save_page(self, url: str, content: PageContent, analysis: dict):
        page_data = {
            "url": url,
            "explorer_tags": analysis["explorer_tags"],
            "abstract": analysis["abstract"],
            "content": content.content,
            "timestamp": datetime.now().isoformat()
        }
        
        filename = f"{self.school}/pages/{hashlib.md5(url.encode()).hexdigest()}.json"
        with open(filename, 'w') as f:
            json.dump(page_data, f, indent=2)
            
        self.page_importance[url] = {
            "score": analysis["importance_score"],
            "tags": analysis["explorer_tags"]
        }

    def save_importance_ranking(self):
        ranked_pages = sorted(
            self.page_importance.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        ranking = [
            {
                "url": url,
                "explorer_tags": data["tags"]
            }
            for url, data in ranked_pages
        ]
        
        with open(f"{self.school}/page_importance_ranking.json", 'w') as f:
            json.dump(ranking, f, indent=2)

    def explore(self):
        to_explore = [self.start_url]
        pages_explored = 0
        
        while to_explore and pages_explored < self.MAX_PAGES:
            current_url = to_explore.pop(0)
            if current_url in self.visited:
                continue
                
            self.logger.info(f"Exploring: {current_url}")
            content = self.fetch_page(current_url)
            if not content:
                continue
                
            self.visited.add(current_url)
            analysis = self.analyze_page(content)
            
            if analysis["importance_score"] > 0.3:
                self.save_page(current_url, content, analysis)
                for link in analysis["recommended_links"]:
                    if link["priority"] in ["high", "medium"]:
                        to_explore.append(link["url"])
                        
            pages_explored += 1
            
        self.save_importance_ranking()
        self.logger.info(f"Exploration complete. {pages_explored} pages analyzed.")
