import click
from llama_index.core.settings import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from .explorer import Explorer
from .beagle import Beagle
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure LlamaIndex settings
Settings.chunk_size = 1024
Settings.chunk_overlap = 20
Settings.num_output = 512

@click.command()
@click.option('--school', prompt='Select medical school', type=click.Choice(['UPenn']), 
              help='Medical school to explore')
@click.option('--url', prompt='Starting URL', 
              default='https://www.med.upenn.edu/admissions/',
              help='URL where exploration should begin')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def main(school: str, url: str, verbose: bool):
    """Medical School Explorer powered by LlamaIndex"""
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Configure logging
    if verbose:
        logging.basicConfig(level=logging.INFO)
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        Settings.callback_manager = callback_manager
        
    click.echo(f"\nStarting LlamaIndex-powered exploration of {school} from {url}")
    
    # Run Explorer
    explorer = Explorer(school, url)
    explorer.explore()
    
    # Run Beagle analysis
    ranking_path = f"{school}/page_importance_ranking.json"
    if os.path.exists(ranking_path):
        beagle = Beagle(school, ranking_path)
        beagle.analyze()
    
if __name__ == '__main__':
    main()
