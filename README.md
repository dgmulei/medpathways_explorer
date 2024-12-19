# MedEx - Medical School Explorer

## Project Vision
MedEx is an AI-powered exploration system designed to democratize access to medical school program information. It aims to replicate and scale the deep analytical work typically performed by experienced pre-med advisors, making comprehensive program insights accessible to all pre-med students.

## Core Components

### 1. Explorer (Verazzano)
- Initial reconnaissance powered by GPT-4o-mini
- Maps medical school website architecture
- Makes rapid relevance assessments
- Identifies high-value content areas
- Generates detailed importance rankings
- Creates initial content annotations

### 2. Analyst (Beagle)
- Deep analysis powered by GPT-4o
- Examines high-priority pages in detail
- Captures both explicit content and implicit signals
- Considers program culture and "fit" indicators
- Prepares content for vector database storage
- Creates rich semantic annotations

## Key Features

### Intelligent Content Discovery
- Domain-aware crawling
- Smart prioritization of relevant content
- Preservation of content hierarchy and context
- Recognition of implicit program signals

### Pre-med Focused Analysis
- Assessment of program requirements
- Evaluation of curriculum details
- Analysis of program culture and values
- Identification of unique offerings
- Documentation of student support resources

### Data Architecture
- Individual page JSONs with full content
- Importance-ranked content index
- Vector-ready semantic chunks
- Rich metadata and advisor insights
- Clear provenance tracking

## Output Structure

### Page Content
```json
{
    "url": "page_url",
    "explorer_tags": ["admissions", "requirements", etc],
    "abstract": "Pre-med focused summary",
    "content": "Full page content",
    "timestamp": "exploration_time"
}
```

### Analysis Output
```json
{
    "sections": [
        {
            "text": "verbatim content",
            "type": "category tag",
            "context": "presentation context",
            "advisor_notes": "significance insights"
        }
    ],
    "program_personality": {
        "tone": "content presentation style",
        "emphasis": "priority areas",
        "distinctive": "unique elements"
    },
    "fit_indicators": [
        "student fit elements"
    ]
}
```

## Setup & Usage

### Requirements
- Python 3.9+
- OpenAI API key
- Playwright for web rendering

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install
```

### Configuration
Create `.env` file with:
```
OPENAI_API_KEY=your-key-here
```

### Running
```bash
python -m medex.cli
```

## Architecture Notes

### Explorer Design
- Focuses on efficient discovery and initial assessment
- Uses rate-limited GPT-4o-mini for quick analysis
- Maintains exploration state and importance ranking
- Generates structured page data

### Beagle Design
- Performs deep content analysis
- Captures page rendering and structure
- Considers visual hierarchy and emphasis
- Creates vector-ready semantic chunks
- Preserves advisor-style insights

## Future Development

### Phase 1: Enhanced Content Understanding
- Improved program culture analysis
- Better fit indicator detection
- More nuanced semantic chunking

### Phase 2: Data Processing
- Integration with vector database
- Enhanced metadata generation
- Structured taxonomy mapping

### Phase 3: Access Layer
- Query interface development
- Semantic search capabilities
- Pre-med focused retrieval

## Contributing
This project is part of a larger initiative to improve pre-med advising access. Contributions that align with this mission are welcome.

## License
Proprietary - All rights reserved