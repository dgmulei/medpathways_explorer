from pathlib import Path
import json
from llama_index.core import Document
from medex.beagle import Beagle

# Initialize Beagle
beagle = Beagle("UPenn", "UPenn/page_importance_ranking.json")

# Process each page file directly
pages_dir = Path("UPenn/pages")
page_files = list(pages_dir.glob("*.json"))
print(f"Found {len(page_files)} pages to analyze")

for i, page_file in enumerate(page_files, 1):
    print(f"\nProcessing page {i}/{len(page_files)}: {page_file.name}")
    with open(page_file) as f:
        page_data = json.load(f)
        
    # Create document from page content
    document = Document(
        text=page_data["content"],
        metadata={"url": page_data["url"]}
    )
    
    # Analyze page
    analysis = beagle.analyze_page(document)
    if analysis:
        nodes = beagle.prepare_nodes(analysis, document)
        
        # Save analysis
        output = {
            "url": page_data["url"],
            "analysis": analysis,
            "nodes": [
                {
                    "text": node.text,
                    "metadata": node.metadata
                } for node in nodes
            ],
            "timestamp": page_data["timestamp"]
        }
        
        output_path = f"UPenn/analysis/{page_file.stem}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
