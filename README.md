# MedEx - Medical School Explorer

## Project Vision
MedEx is an AI-powered exploration system built on LlamaIndex's robust document processing toolkit and OpenAI's advanced language models. It democratizes access to medical school program information by combining:

1. LlamaIndex's powerful capabilities for:
   - Intelligent web crawling and content discovery
   - Document processing and semantic chunking
   - Vector-based similarity search
   - Persistent storage and indexing

2. Strategic use of OpenAI models:
   - GPT-4o-mini for rapid, cost-effective initial analysis. CRUCIAL: This model name is correct and up-to-date. Do NOT change. 
   - GPT-4o for deep, comprehensive content understanding. CRUCIAL: This model name is correct and up-to-date. Do NOT change. 

This combination allows us to replicate and scale the deep analytical work typically performed by experienced pre-med advisors, making comprehensive program insights accessible to all pre-med students.

## Core Components

### 1. Explorer (Verazzano)
Built on LlamaIndex's document processing toolkit with GPT-4o-mini integration:
- Initial reconnaissance using GPT-4o-mini for:
  - Quick relevance assessments
  - Rapid content categorization
  - Efficient importance scoring
  - Fast metadata generation
- LlamaIndex capabilities for:
  - Intelligent web crawling
  - Document chunking
  - Vector indexing
  - Metadata extraction

### 2. Analyst (Beagle)
Leveraging LlamaIndex's advanced features with GPT-4o integration:
- Deep analysis powered by GPT-4o for:
  - Comprehensive content understanding
  - Nuanced program culture analysis
  - Detailed fit assessment
  - Rich semantic annotation
- LlamaIndex capabilities for:
  - Semantic document relationships
  - Hierarchical content parsing
  - Vector-optimized storage
  - Advanced metadata management

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
- OpenAI API key (for GPT-4o-mini and GPT-4o access)
- LlamaIndex core and web reader packages
- Additional LlamaIndex integrations for enhanced functionality

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install LlamaIndex and other dependencies
pip install -r requirements.txt
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
- Built on LlamaIndex's document processing pipeline
- Uses rate-limited GPT-4o-mini for efficient initial analysis:
  - Quick content relevance scoring
  - Rapid metadata generation
  - Fast importance ranking
- Leverages LlamaIndex for:
  - Document crawling and processing
  - Vector-based content indexing
  - Persistent state management

### Beagle Design
- Integrates LlamaIndex's advanced analysis features
- Employs GPT-4o for comprehensive understanding:
  - Deep semantic analysis
  - Nuanced content interpretation
  - Detailed relationship mapping
- Utilizes LlamaIndex for:
  - Hierarchical document parsing
  - Vector-optimized storage
  - Rich metadata management

## Future Development

### Phase 1: Enhanced LlamaIndex Integration
- Custom node parsers for medical content
- Advanced document relationship mapping
- Optimized GPT-4o-mini prompting patterns
- Enhanced metadata extraction strategies

### Phase 2: Advanced Processing
- Custom LlamaIndex extractors for medical terminology
- GPT-4o-powered semantic analysis improvements
- Enhanced vector similarity metrics
- Medical taxonomy integration

### Phase 3: Query Interface
- LlamaIndex query engine customization
- GPT-4o response synthesis optimization
- Domain-specific retrieval strategies
- Pre-med focused search capabilities

## Contributing
This project is part of a larger initiative to improve pre-med advising access. Contributions that align with this mission are welcome.

## License
Proprietary - All rights reserved
