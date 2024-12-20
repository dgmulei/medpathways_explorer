from llama_index.core import Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import AsyncWebPageReader
import asyncio
from typing import Dict, List, Optional
from urllib.parse import urlparse
import json
from datetime import datetime
import os
import hashlib
from .utils import setup_logging

# Configure LlamaIndex settings
Settings.chunk_size = 1024
Settings.chunk_overlap = 20
Settings.num_output = 512  # Limit response length for efficiency

class Explorer:
    def __init__(self, school: str, start_url: str):
        self.school = school
        self.start_url = start_url
        self.base_domain = urlparse(start_url).netloc
        self.page_importance = {}
        # Setup LlamaIndex debug handler
        self.llama_debug = LlamaDebugHandler()
        callback_manager = CallbackManager([self.llama_debug])
        
        # Setup LlamaIndex settings with GPT-4o-mini
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_version="2024-02"
        )
        Settings.callback_manager = callback_manager
        
        self.MAX_PAGES = 500
        
        # LlamaIndex setup for web page reading with async support
        self.loader = AsyncWebPageReader(
            html_to_text=True,  # Convert HTML to clean text for better parsing
            limit=5,  # Limit concurrent requests to be respectful
            dedupe=True,  # Remove duplicate URLs
            fail_on_error=False  # Continue on errors to be resilient
        )
        
        # Setup node parser with default configuration
        self.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        
        # Setup storage
        self.storage_dir = f"{school}/index_storage"
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Create new storage context
        self.storage_context = StorageContext.from_defaults()
        
        self.logger = setup_logging(school, "explorer")
        os.makedirs(f"{school}/pages", exist_ok=True)
        
    def fetch_pages(self) -> List[Document]:
        try:
            # Load documents with async support
            documents = asyncio.run(self.loader.aload_data(urls=[self.start_url]))
            for doc in documents:
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata["url"] = self.start_url
            self.logger.info(f"Loaded {len(documents)} documents")
            
            # Process documents into nodes
            nodes = self.node_parser.get_nodes_from_documents(documents)
            self.logger.info(f"Created {len(nodes)} nodes")
            
            # Create and persist vector index
            self.index = VectorStoreIndex(
                nodes,
                storage_context=self.storage_context
            )
            # Persist storage after indexing
            self.index.storage_context.persist(persist_dir=self.storage_dir)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error fetching pages: {str(e)}")
            return []

    def analyze_page(self, document: Document) -> dict:
        
        try:
            # Create analysis prompt
            # Log the raw content for debugging
            self.logger.info(f"Raw content length: {len(document.text)}")
            self.logger.info(f"Content sample: {document.text[:500]}")
            
            prompt = f"""You are an experienced pre-med advisor analyzing a medical school webpage. Your task is to discover ALL content paths valuable to pre-med students by thinking like an advisor who has reviewed hundreds of medical school websites.

CRITICAL: Your primary task is to discover ALL possible navigation paths and content references. You must:
1. Identify explicit links in the content (URLs, paths)
2. Recognize implicit references to other pages
3. Understand the navigation structure
4. Find mentions of related content

BE EXTREMELY THOROUGH in link discovery:
- Look for navigation menu items (e.g., "Admissions", "Requirements")
- Find section references (e.g., "Visit our Requirements page")
- Identify related content mentions (e.g., "Learn more about our curriculum")
- Spot application portal links (e.g., "Apply Now", "Submit Application")
- Detect resource download links (e.g., "Download PDF", "View Guide")
- Notice program requirement pages (e.g., "Prerequisites", "MCAT Requirements")
            
CONTENT PRIORITIES:
1. Core Pre-med Information:
   - Admissions requirements and competencies
   - Application processes and deadlines
   - Curriculum structure and unique features
   - Financial information and opportunities
   - Student support and resources
   - Program culture and values

2. Navigation Recognition:
   - Main navigation menus
   - Section headers and submenus
   - Related content references
   - Resource collections
   - Application portals and tools

3. Link Discovery Patterns:
   - Direct menu/navigation links
   - In-content references ("learn more about X")
   - Related resource mentions
   - Important document links (PDFs, guides)
   - Contact points and portals

4. Content Value Signals:
   - Direct applicant guidance
   - Program requirements
   - Application instructions
   - Student support information
   - Unique program features
   - Decision-critical content

Return a JSON object with these EXACT keys and format (no additional text):
{{
    "importance_score": 0.9,
    "explorer_tags": ["admissions", "requirements"],
    "abstract": "Brief summary of the page content",
    "recommended_links": [
        {{
            "url": "/admissions/requirements.html",
            "text": "Admissions Requirements",
            "type": "navigation",
            "priority": 0.9,
            "source": "Main Navigation Menu",
            "confidence": 1.0
        }}
    ],
    "related_topics": ["admissions process", "requirements"]
}}

IMPORTANT: Follow this EXACT format. Do not include any explanatory text in the JSON.

For each section of content, analyze it like an advisor would:
1. Look for explicit URLs or paths (e.g., "/admissions/requirements")
2. Identify navigation elements (e.g., menu items, breadcrumbs)
3. Find content references (e.g., "See our curriculum guide")
4. Note resources (e.g., "Download application checklist")
5. Consider student navigation needs (e.g., "What would they click next?")

BE AGGRESSIVE in identifying potential links - if there's any mention of other content or pages, include it as a recommended link.

Content to analyze:
{document.text[:2000]}
            """
            
            # Extract and process links from raw content
            import re
            from urllib.parse import urljoin
            
            # Get base URL from current URL
            base_url = '/'.join(self.start_url.split('/')[:-1]) + '/'
            
            # Extract markdown-style links
            raw_links = re.findall(r'\[(.*?)\]\((.*?)\)', document.text)
            extracted_links = []
            
            for link_text, link_url in raw_links:
                if link_url.endswith('.html'):  # Only include HTML pages
                    # Clean up link text and make URL absolute
                    text = link_text.replace('__', '').strip()
                    url = urljoin(base_url, link_url)
                    
                    extracted_links.append({
                        "url": url,
                        "text": text,
                        "type": "navigation",
                        "priority": 0.8,
                        "source": "Navigation Menu",
                        "confidence": 1.0
                    })
            
            # Log extracted links for debugging
            self.logger.info(f"Extracted {len(extracted_links)} raw links")
            for link in extracted_links[:5]:
                self.logger.info(f"Raw link: {link}")
            
            # Use query engine for analysis
            response = self.index.as_query_engine(
                verbose=True
            ).query(prompt)
            
            try:
                # Try to parse as JSON first
                result = json.loads(str(response))
                # Add extracted links to the recommended_links
                if "recommended_links" not in result:
                    result["recommended_links"] = []
                result["recommended_links"].extend(extracted_links)
            except json.JSONDecodeError:
                # If not valid JSON, create structured response
                result = {
                    "importance_score": 0.8,  # Default high for admissions page
                    "explorer_tags": ["admissions", "requirements"],
                    "abstract": str(response)[:100],  # Use first 100 chars as abstract
                    "recommended_links": extracted_links,  # Use extracted links
                    "related_topics": []
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {document.metadata.get('url', '')}: {str(e)}")
            return {
                "importance_score": 0,
                "explorer_tags": [],
                "abstract": "",
                "recommended_links": [],
                "related_topics": []
            }

    def save_page(self, url: str, document: Document, analysis: dict):
        page_data = {
            "url": url,
            "explorer_tags": analysis["explorer_tags"],
            "abstract": analysis["abstract"],
            "content": document.text,
            "related_topics": analysis.get("related_topics", []),
            "metadata": {
                "title": document.metadata.get("title", ""),
                "links": document.metadata.get("links", []),
                "importance_score": analysis["importance_score"],
                "recommended_links": analysis["recommended_links"],
                "source_metadata": document.metadata
            },
            "timestamp": datetime.now().isoformat()
        }
        
        filename = f"{self.school}/pages/{hashlib.md5(url.encode()).hexdigest()}.json"
        with open(filename, 'w') as f:
            json.dump(page_data, f, indent=2)
            
        self.page_importance[url] = {
            "score": analysis["importance_score"],
            "tags": analysis["explorer_tags"],
            "related_topics": analysis.get("related_topics", [])
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
                "explorer_tags": data["tags"],
                "importance_score": data["score"],
                "related_topics": data["related_topics"],
                "semantic_context": {
                    "related_pages": [
                        topic for topic in data["related_topics"]
                        if any(url_part in topic.lower() for url_part in ["admission", "requirement", "curriculum"])
                    ],
                    "topic_clusters": list(set(
                        tag for tags in [d["tags"] for d in self.page_importance.values()]
                        for tag in tags
                    ))
                }
            }
            for url, data in ranked_pages
        ]
        
        with open(f"{self.school}/page_importance_ranking.json", 'w') as f:
            json.dump({
                "ranking": ranking,
                "metadata": {
                    "total_pages": len(self.page_importance),
                    "exploration_timestamp": datetime.now().isoformat(),
                    "base_domain": self.base_domain,
                    "topic_overview": list(set(
                        topic for data in self.page_importance.values()
                        for topic in data.get("related_topics", [])
                    ))
                }
            }, f, indent=2)

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is nested under the starting URL path"""
        try:
            parsed = urlparse(url)
            start_parsed = urlparse(self.start_url)
            # Must be same domain
            if self.base_domain != parsed.netloc:
                return False
            # URL path must start with the starting URL's path
            if not parsed.path.startswith(start_parsed.path):
                return False
            return True
        except:
            return False

    def explore(self):
        self.logger.info(f"Starting exploration from: {self.start_url}")
        
        # Initialize URL tracking
        urls_to_visit = {self.start_url}
        visited_urls = set()
        
        while urls_to_visit and len(visited_urls) < self.MAX_PAGES:
            current_url = urls_to_visit.pop()
            if current_url in visited_urls:
                continue
                
            self.logger.info(f"Processing: {current_url}")
            visited_urls.add(current_url)
            
            # Fetch and process page with async support
            documents = asyncio.run(self.loader.aload_data(urls=[current_url]))
            if not documents:
                continue
                
            # Use the first document as the main page
            document = documents[0]
            
            # Collect all unique links from all documents
            all_links = set()
            for doc in documents:
                if doc.metadata and "links" in doc.metadata:
                    all_links.update(doc.metadata["links"])
            document.metadata["links"] = list(all_links)
            if not document.metadata:
                document.metadata = {}
            document.metadata["url"] = current_url
            
            # Process document into nodes
            nodes = self.node_parser.get_nodes_from_documents([document])
            
            # Update vector index
            self.index = VectorStoreIndex(
                nodes,
                storage_context=self.storage_context
            )
            
            # Analyze page
            analysis = self.analyze_page(document)
            
            if analysis["importance_score"] > 0.3:
                self.save_page(current_url, document, analysis)
                
                # Add new URLs to visit from analysis
                for link in analysis.get("recommended_links", []):
                    if isinstance(link, dict):
                        url = link.get("url", "")
                        priority = link.get("priority", 0)
                        link_type = link.get("type", "")
                        # Prioritize navigation and content links
                        if (url and priority > 0.3 and 
                            self.is_valid_url(url) and
                            link_type in ["navigation", "content", "application"]):
                            urls_to_visit.add(url)
                    elif isinstance(link, str) and self.is_valid_url(link):
                        urls_to_visit.add(link)
            
            # Log discovered links
            links = analysis.get("recommended_links", [])
            self.logger.info(f"Found {len(links)} links in {current_url}")
            for link in links[:5]:  # Log first 5 links
                if isinstance(link, dict):
                    self.logger.info(f"Link: {link.get('url', '')} - {link.get('text', '')[:50]}")
            
            # Persist storage periodically
            if len(visited_urls) % 5 == 0:
                self.index.storage_context.persist(persist_dir=self.storage_dir)
        
        self.save_importance_ranking()
        self.logger.info(f"Exploration complete. {len(visited_urls)} pages analyzed.")
