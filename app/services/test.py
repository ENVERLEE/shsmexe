import asyncio
from dotenv import load_dotenv
import json
import os
from perplexity_service import PerplexityService, SearchConfig  # Replace 'your_module_name' with the actual module name

# Load environment variables
load_dotenv()

async def main():
    api_key = os.environ.get("PERPLEXITY_API_KEY")  # Assuming you've named the env variable like this
    if not api_key:
        raise ValueError("API key not found in environment variables")

    service = PerplexityService(api_key)
    
    # Example query
    query = "quantum computing advancements"
    config = SearchConfig(search_domain="academic", search_recency_filter="year")
    
    try:
        results = await service.search_references(query, config=config)
        print(results)
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())