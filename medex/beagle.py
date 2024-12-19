from llama_index.core import Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import SimpleWebPageReader
from typing import List, Dict, Optional
import json
import os
import hashlib
import time
from datetime import datetime
from .utils import setup_logging

class Beagle:
    def __init__(self, school: str, importance_ranking_path: str):
        self.school = school
        
        # Setup LlamaIndex components
        self.llama_debug = LlamaDebugHandler()
        callback_manager = CallbackManager([self.llama_debug])
        
        # Use GPT-4o for deeper analysis
        Settings.llm = OpenAI(
            model="gpt-4o",
            temperature=0,
            api_version="2024-02"
        )
        Settings.callback_manager = callback_manager
        
        # Initialize node parser
        self.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        
        # Initialize web page reader
        self.reader = SimpleWebPageReader()
        
        # Load pages to analyze
        with open(importance_ranking_path) as f:
            data = json.load(f)
            self.pages_to_analyze = data.get('ranking', [])
            
        self.logger = setup_logging(school, "beagle")
        os.makedirs(f"{school}/analysis", exist_ok=True)

    def capture_page(self, url: str) -> Optional[Document]:
        try:
            # Use LlamaIndex's web page reader
            documents = self.reader.load_data([url])
            
            if not documents:
                return None
                
            # Process document with HTML parser
            nodes = self.node_parser.get_nodes_from_documents(documents)
            
            # Create temporary index for this page
            storage_context = StorageContext.from_defaults()
            index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                verbose=True
            )
            
            return documents[0]
            
        except Exception as e:
            self.logger.error(f"Failed to capture {url}: {str(e)}")
            return None

    def analyze_page(self, document: Document) -> Optional[Dict]:
        try:
            # Use LlamaIndex's query engine for content analysis
            query_engine = VectorStoreIndex([document]).as_query_engine(
                verbose=True
            )
            
            # Create analysis prompt
            prompt = f"""You are analyzing {self.school}'s medical school webpage.
            Consider both content and presentation to provide insights for pre-med students.
            
            You must respond with a valid JSON object using this exact structure, with no additional text or explanation:
            {{
                "sections": [
                    {{
                        "text": "exact content from page",
                        "type": "content category",
                        "context": "presentation details",
                        "advisor_notes": "insights for students"
                    }}
                ],
                "program_personality": {{
                    "tone": "content style",
                    "emphasis": "key priorities",
                    "distinctive": "unique features"
                }},
                "fit_indicators": [
                    "student fit factors"
                ]
            }}

            Content to analyze:
            {document.text[:2000]}
            """
            
            # Use query engine for analysis
            response = query_engine.query(prompt)
            response_text = str(response)
            
            try:
                # Log the response for debugging
                self.logger.info(f"Raw response: {response_text[:200]}...")
                
                # Clean up the response text
                if response_text.strip():
                    # Remove markdown code blocks if present
                    if "```" in response_text:
                        # Split by ``` and take the content between the markers
                        parts = response_text.split("```")
                        if len(parts) >= 2:
                            response_text = parts[1]
                            # Remove "json" if it's at the start
                            if response_text.startswith("json"):
                                response_text = response_text[4:]
                            # Remove any trailing ```
                            response_text = response_text.split("```")[0]
                    
                    # Try to parse as JSON
                    return json.loads(response_text.strip())
                else:
                    self.logger.error("Empty response received")
                    return None
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON response: {str(e)}")
                self.logger.error(f"Response text: {response_text[:200]}...")
                return None
            
        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}")
            return None

    def prepare_nodes(self, analysis: Dict, document: Document) -> List[Document]:
        if not analysis or "sections" not in analysis:
            return []
            
        nodes = []
        for section in analysis['sections']:
            # Create Document for each section with rich metadata
            node = Document(
                text=section['text'],
                metadata={
                    "type": section['type'],
                    "context": section['context'],
                    "advisor_notes": section['advisor_notes'],
                    "url": document.metadata.get("url", ""),
                    "source": document.metadata
                }
            )
            nodes.append(node)
        return nodes

    def analyze(self):
        for page in self.pages_to_analyze:
            try:
                url = page['url']
                self.logger.info(f"Analyzing {url}")
                
                document = self.capture_page(url)
                if not document:
                    continue
                    
                analysis = self.analyze_page(document)
                if not analysis:
                    continue
                    
                nodes = self.prepare_nodes(analysis, document)
                
                output = {
                    "url": url,
                    "analysis": analysis,
                    "nodes": [
                        {
                            "text": node.text,
                            "metadata": node.metadata
                        } for node in nodes
                    ],
                    "timestamp": datetime.now().isoformat()
                }
                
                filename = f"{self.school}/analysis/{hashlib.md5(url.encode()).hexdigest()}.json"
                with open(filename, 'w') as f:
                    json.dump(output, f, indent=2)
                    
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error processing {page['url']}: {str(e)}")
                continue
