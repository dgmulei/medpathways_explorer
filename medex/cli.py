import click
from .explorer import Explorer
from .beagle import Beagle
import os
from dotenv import load_dotenv

load_dotenv()

@click.command()
@click.option('--school', prompt='Select medical school', type=click.Choice(['UPenn']), 
              help='Medical school to explore')
@click.option('--url', prompt='Starting URL', 
              default='https://www.med.upenn.edu/admissions/',
              help='URL where exploration should begin')
def main(school: str, url: str):
    """Medical School Explorer"""
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
        
    click.echo(f"\nStarting exploration of {school} from {url}")
    
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