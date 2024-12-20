import json
import os
from pathlib import Path
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# New enhanced prompt
ENHANCED_PROMPT = """"You are analyzing UPenn's medical school webpage. Your task is to extract and analyze ALL content from the main body of the page.

PAGE VALIDATION:
First check if page is valid:
1. Check <title> and <h2> for "Page Not Found"
2. Look for error messages in content area
3. Verify presence of actual content in main article
DO NOT PROCEED with analysis if page is 404/error.

CONTENT CHUNKING RULES:
1. Page Size Detection:
   - Check content length before processing
   - If > 6000 tokens, split into logical sections
   - Process each section independently
   - Merge results maintaining JSON structure

2. Section Boundaries:
   - Split at major headers (h2)
   - Keep related content together
   - Maintain context between chunks
   - Preserve data point relationships

3. Content Priority:
   - Process critical sections first
   - Include key headers in each chunk
   - Ensure requirements span chunks
   - Track cross-references

CONTENT LOCATION:
1. Primary Content Path (MUST CHECK FIRST):
   <article class="content main row subpage">
     <section class="grid">
       <section class="unit">
         [THIS IS THE MAIN CONTENT]
       </section>
     </section>
   </article>

2. Secondary Content Paths (check if primary empty):
   - Direct <main> content
   - Content between breadcrumb and footer
   - Areas with classes containing 'content', 'main', 'subpage'

3. Special Content Areas:
   - Content within <!--content area--> comments
   - Data tables with class attributes
   - Lists within <section class="unit">
   - Content following headers (h2-h4)

EXPLICITLY IGNORE:
1. Meta Content:
   - <meta> descriptions
   - Navigation menus
   - Headers/footers
   - Mission statements
   - Contact sections (unless contact page)

2. Error Content:
   - 404 messages
   - Error notifications
   - "Page not found" text
   - Empty content areas

EXTRACTION RULES:
1. Content Hierarchy:
   - Preserve header relationships (h2 > h3 > h4)
   - Maintain list structures (ul/ol)
   - Keep table formats intact
   - Respect content groupings

2. Data Capture:
   - Extract ALL text from valid sections
   - Preserve formatting and structure
   - Include complete lists/tables
   - Maintain content relationships

3. Special Elements:
   - Capture dates in MM/DD/YYYY format
   - Extract numerical data points
   - Identify requirements lists
   - Note application deadlines
   - Record contact information

JSON SAFETY RULES:
1. Text Content:
   - Maximum 1000 characters per section
   - Split longer content into multiple sections
   - Escape special characters (quotes, newlines)
   - Remove HTML tags from extracted text

2. Data Points:
   - Keep arrays and values concise
   - Use simple string formats
   - Avoid nested structures
   - Escape special characters

3. Section Types:
   - Use consistent type names
   - Keep context paths simple
   - Avoid special characters in keys
   - Use clear category names

4. JSON Output Rules:
   - Keep it minimal and flat
   - Use basic strings only
   - No optional fields
   - Example format:
     {
       "sections": [
         {
           "text": "content here",
           "type": "category",
           "data_points": {
             "key": ["value"]
           }
         }
       ]
     }

   IMPORTANT: No extra fields, no nested objects, no complex formatting.

You must respond with a valid JSON object using this exact structure:
{
    "sections": [
        {
            "text": "content (max 1000 chars, escaped)",
            "type": "content category (simple string)",
            "context": "location (simple path)",
            "data_points": {
                "dates": ["MM/DD/YYYY only"],
                "requirements": ["simple strings"],
                "contacts": ["basic contact info"],
                "steps": ["numbered steps"],
                "statistics": ["number: description"],
                "resources": ["url or name"]
            },
            "advisor_notes": "brief insights"
        }
    ],
    "program_personality": {
        "tone": "simple description",
        "emphasis": "key focus",
        "distinctive": "unique aspects"
    },
    "fit_indicators": [
        "simple factors"
    ]
}

Content to analyze:
"""

def analyze_with_new_prompt(html_content):
    """Analyze HTML content using the enhanced prompt"""
    try:
        # Check for 404 page
        if '"Page Not Found"' in html_content or '<h2>Page Not Found</h2>' in html_content:
            print("Skipping 404 page")
            return {
                "sections": [{
                    "text": "Page Not Found",
                    "type": "error",
                    "context": "404 error page",
                    "data_points": {}
                }]
            }

        # Clean HTML content
        html_content = html_content.replace('\n', ' ').replace('\r', ' ')
        
        # Estimate token count (rough estimate: 4 chars = 1 token)
        estimated_tokens = len(html_content) / 4
        
        if estimated_tokens > 6000:
            print(f"Large content detected ({estimated_tokens:.0f} estimated tokens). Splitting into chunks.")
            # For now, just take the first 6000 tokens worth to test chunking logic
            html_content = html_content[:24000]  # 24000 chars ≈ 6000 tokens
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": ENHANCED_PROMPT},
                {"role": "user", "content": html_content}
            ],
            temperature=0
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print("Raw response:", response.choices[0].message.content[:200])
            return None
            
    except Exception as e:
        print(f"Error analyzing content: {str(e)}")
        return None

def compare_analyses(old_analysis, new_analysis):
    """Compare old and new analyses to highlight differences"""
    differences = {
        "content_depth": len(str(new_analysis)) - len(str(old_analysis)),
        "section_count": len(new_analysis.get("sections", [])) - len(old_analysis.get("sections", [])),
        "has_data_points": "data_points" in new_analysis.get("sections", [{}])[0],
        "meta_only": all("meta" in s.get("context", "").lower() for s in old_analysis.get("sections", [])),
        "old_sections": [{
            "type": s.get("type"),
            "context": s.get("context"),
            "text_preview": s.get("text", "")[:100] + "..."
        } for s in old_analysis.get("sections", [])],
        "new_sections": [{
            "type": s.get("type"),
            "context": s.get("context"),
            "text_preview": s.get("text", "")[:100] + "...",
            "data_points": {k: v for k, v in s.get("data_points", {}).items() if v}
        } for s in new_analysis.get("sections", [])]
    }
    return differences

def test_prompt():
    """Test the enhanced prompt on existing pages"""
    pages_dir = Path("UPenn/pages")
    analyses_dir = Path("UPenn/analysis")
    output_dir = Path("UPenn/prompt_tests")
    output_dir.mkdir(exist_ok=True)
    results = []

    # Test final pages
    test_files = [
        "88c4f4b44e10f87f6e628b9d50c51ae2.json",  # Core curriculum page
        "461b5aa786b8dd92eb0199a8d10deecd.json"   # Application timeline page
    ]

    for filename in test_files:
        page_path = pages_dir / filename
        analysis_path = analyses_dir / filename
        
        if not page_path.exists() or not analysis_path.exists():
            continue

        print(f"\nTesting {filename}...")
        
        # Load files
        with open(page_path) as f:
            page_data = json.load(f)
        with open(analysis_path) as f:
            old_analysis = json.load(f)

        # Run new analysis
        new_analysis = analyze_with_new_prompt(page_data["content"])
        if not new_analysis:
            continue

        # Save new analysis
        if new_analysis:
            output_path = output_dir / f"test_{filename}"
            with open(output_path, 'w') as f:
                json.dump({
                    "url": page_data.get("url", ""),
                    "analysis": new_analysis,
                    "timestamp": page_data.get("timestamp", "")
                }, f, indent=2)
            print(f"\nSaved test results to: {output_path}")

        # Compare results
        differences = compare_analyses(old_analysis.get("analysis", {}), new_analysis)
        
        results.append({
            "file": filename,
            "differences": differences,
            "improved": differences["content_depth"] > 0 and differences["has_data_points"]
        })

        print(f"\nResults for {filename}:")
        print("=" * 80)
        print("OLD ANALYSIS:")
        for section in differences["old_sections"]:
            print(f"\nType: {section['type']}")
            print(f"Context: {section['context']}")
            print(f"Preview: {section['text_preview']}")
        
        print("\nNEW ANALYSIS:")
        for section in differences["new_sections"]:
            print(f"\nType: {section['type']}")
            print(f"Context: {section['context']}")
            print(f"Preview: {section['text_preview']}")
            if section.get("data_points"):
                print("Data Points:")
                for k, v in section["data_points"].items():
                    if v:
                        print(f"  {k}: {v}")
        
        print("\nSUMMARY:")
        print(f"Content depth change: {differences['content_depth']} characters")
        print(f"Section count change: {differences['section_count']}")
        print(f"Has data points: {differences['has_data_points']}")
        print(f"Old analysis was meta-only: {differences['meta_only']}")

    return results

if __name__ == "__main__":
    print("Testing enhanced prompt...")
    results = test_prompt()
    print("\nOverall Results:")
    improved_count = sum(1 for r in results if r["improved"])
    print(f"{improved_count}/{len(results)} pages showed improvement")
