from llama_index.core import Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import SimpleWebPageReader
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
        
        # LlamaIndex setup for web page reading
        self.loader = SimpleWebPageReader()
        
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
            # Load documents with metadata
            documents = self.loader.load_data([self.start_url])
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
            prompt = f"""You are an experienced pre-med advisor analyzing a medical school webpage. Your task is to discover ALL content paths valuable to pre-med students by thinking like an advisor who has reviewed hundreds of medical school websites.

CRITICAL: Discover ALL possible navigation paths. Your task is to understand the content and identify every possible linked page or resource. BE AGGRESSIVE in identifying potential links - if there's any mention of other content or pages, include it as a recommended link.
            
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

Return a JSON object with:
- importance_score (0-1): Based on pre-med student relevance
- explorer_tags: Categories like ["admissions", "requirements", "curriculum"]
- abstract: 100-word summary of key information
- recommended_links: Array of discovered content paths with priority scores
- related_topics: Key themes and connections

Think like an advisor guiding students through ANY medical school's content, identifying valuable information regardless of how it's structured or presented.

            BE AGGRESSIVE in identifying potential links - if there's any mention of other content or pages, include it as a recommended link.

            Content to analyze:
            {document.text[:2000]}
            """
            
            # Use query engine for analysis
            response = self.index.as_query_engine(
                verbose=True
            ).query(prompt)
            
            # Parse response into expected format
            try:
                # Try to parse as JSON first
                result = json.loads(str(response))
            except json.JSONDecodeError:
                # If not valid JSON, create structured response
                result = {
                    "importance_score": 0.8,  # Default high for admissions page
                    "explorer_tags": ["admissions", "requirements"],
                    "abstract": str(response)[:100],  # Use first 100 chars as abstract
                    "recommended_links": [],
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
            
            # Fetch and process page
            documents = self.loader.load_data([current_url])
            if not documents:
                continue
                
            document = documents[0]
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
