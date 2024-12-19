from playwright.sync_api import sync_playwright
import openai
from dataclasses import dataclass
from typing import List, Dict
import json
import os
from datetime import datetime
from bs4 import BeautifulSoup
from .utils import setup_logging

@dataclass
class PageView:
    url: str
    html: str
    text_content: str
    structure: Dict
    visual_elements: List[Dict]

class Beagle:
    def __init__(self, school: str, importance_ranking_path: str):
        self.school = school
        with open(importance_ranking_path) as f:
            self.pages_to_analyze = json.load(f)
        self.logger = setup_logging(school, "beagle")
        os.makedirs(f"{school}/analysis", exist_ok=True)

    def capture_page(self, url: str) -> PageView:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, wait_until='networkidle')
            
            structure = page.evaluate("""() => {
                function getNodeStructure(node) {
                    const rect = node.getBoundingClientRect();
                    return {
                        tag: node.tagName,
                        classes: Array.from(node.classList),
                        position: {
                            top: rect.top,
                            left: rect.left,
                            width: rect.width,
                            height: rect.height
                        },
                        children: Array.from(node.children).map(getNodeStructure)
                    };
                }
                return getNodeStructure(document.body);
            }""")
            
            visual_elements = page.evaluate("""() => {
                return Array.from(document.querySelectorAll('table, img, .chart, [role="img"]'))
                    .map(el => ({
                        type: el.tagName.toLowerCase(),
                        position: el.getBoundingClientRect(),
                        content: el.tagName === 'TABLE' ? el.innerText : null,
                        alt: el.alt || null
                    }));
            }""")
            
            html = page.content()
            text_content = page.evaluate('() => document.body.innerText')
            browser.close()
            
            return PageView(
                url=url,
                html=html,
                text_content=text_content,
                structure=structure,
                visual_elements=visual_elements
            )

    def analyze_page(self, page_view: PageView) -> Dict:
        base_prompt = """You are a top pre-med advisor analyzing {school}'s medical school webpage.
        Consider both content and presentation.
        
        Return structured JSON:
        {
            "sections": [
                {
                    "text": "verbatim content",
                    "type": "category tag",
                    "context": "where/how presented",
                    "advisor_notes": "insights on significance/implications"
                }
            ],
            "program_personality": {
                "tone": "how content is presented",
                "emphasis": "what's highlighted/prioritized",
                "distinctive": "unique elements noted"
            },
            "fit_indicators": [
                "specific elements that help assess student fit"
            ]
        }"""

        messages = [
            {"role": "system", "content": base_prompt.format(school=self.school)},
            {"role": "user", "content": json.dumps({
                "url": page_view.url,
                "content": page_view.text_content,
                "layout": {
                    "structure": page_view.structure,
                    "visual_elements": page_view.visual_elements
                }
            })}
        ]

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}")
            return {
                "sections": [],
                "program_personality": {
                    "tone": "Error in analysis",
                    "emphasis": "",
                    "distinctive": ""
                },
                "fit_indicators": []
            }

    def prepare_vectors(self, analysis: Dict) -> List[Dict]:
        chunks = []
        
        for section in analysis['sections']:
            chunk = {
                "content": section['text'],
                "metadata": {
                    "url": analysis['url'],
                    "section_type": section['type'],
                    "context": section['context'],
                    "advisor_notes": section['advisor_notes']
                }
            }
            chunks.append(chunk)

        return chunks

    def analyze(self):
        for page in self.pages_to_analyze:
            try:
                self.logger.info(f"Analyzing {page['url']}")
                
                page_view = self.capture_page(page['url'])
                analysis = self.analyze_page(page_view)
                vectors = self.prepare_vectors(analysis)
                
                output = {
                    "url": page['url'],
                    "analysis": analysis,
                    "vectors": vectors,
                    "timestamp": datetime.now().isoformat()
                }
                
                filename = f"{self.school}/analysis/{hashlib.md5(page['url'].encode()).hexdigest()}.json"
                with open(filename, 'w') as f:
                    json.dump(output, f, indent=2)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {page['url']}: {str(e)}")
                continue