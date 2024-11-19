import ssl
import aiohttp
import asyncio
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Literal
import logging
from dotenv import load_dotenv
import re
import certifi
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Type definitions
RecencyFilter = Literal["day", "week", "month", "year", "all"]
SearchDomain = Literal["academic", "web", "news"]

@dataclass
class SearchConfig:
    """Configuration for search parameters"""
    max_tokens: Optional[int] = None
    temperature: float = 0.2
    top_p: float = 0.9
    return_images: bool = False
    return_related_questions: bool = False
    search_recency_filter: RecencyFilter = "month"
    top_k: int = 0
    stream: bool = False
    presence_penalty: float = 0
    frequency_penalty: float = 1
    search_domain: SearchDomain = "academic"

@dataclass
class AcademicReference:
    """Structure for academic reference data"""
    title: str
    authors: Optional[List[str]]
    abstract: Optional[str]
    publication_date: Optional[str]
    journal: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    citations: Optional[int]

class PerplexityAPIError(Exception):
    """Custom exception for Perplexity API errors"""
    pass

class ValidationError(Exception):
    """Custom exception for input validation errors"""
    pass

class PerplexityService:
    """Service for academic reference search using Perplexity AI API"""
    
    def __init__(self, api_key: str):
        """Initialize PerplexityService with API key and configuration"""
        if not api_key:
            raise ValueError("API key cannot be empty")
            
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.logger = logging.getLogger(__name__)
        
        # Initialize SSL context
        self._init_ssl_context()
        
        # Default system prompt
        self.system_prompt = """
        You are an advanced academic search assistant specialized in retrieving and formatting scientific paper information. 
        Your task is to process a user's query within a specific field of study and return relevant paper information in a structured JSON format.

        Before providing the final output, outline your search and analysis process inside <search_strategy> tags. Consider the following steps:

        1. Break down the user's query into key components within the context of the given specialization.
        2. List potential search terms and filters derived from the query and specialization.
        3. Outline your strategy for finding relevant papers, including databases or sources you would use.
        4. Describe your approach for evaluating paper relevance and importance.
        5. For each paper you would find, plan how to provide a comprehensive abstract.
        6. Detail your method for collecting and formatting the required information for each paper.

        After your search strategy, provide the search results in the specified JSON format:

        {
            "references": [{
                "title": "Paper title",
                "authors": ["Author 1", "Author 2"],
                "abstract": "Paper abstract",
                "publication_date": "YYYY-MM-DD",
                "journal": "Journal name",
                "doi": "DOI number",
                "url": "Paper URL",
                "citations": integer_count
            }],
            "metadata": {
                "total_results": integer,
                "search_timestamp": "ISO datetime",
                "query_context": "search context"
            }
        }
        """

    def _init_ssl_context(self):
        """Initialize and configure SSL context"""
        try:
            if not os.path.exists(certifi.where()):
                raise PerplexityAPIError("SSL certificate file not found")
            
            self.ssl_context = ssl.create_default_context(cafile=certifi.where())
            self.ssl_context.check_hostname = True
            self.ssl_context.verify_mode = ssl.CERT_REQUIRED
            
            # Additional SSL options
            self.ssl_context.options |= ssl.OP_NO_SSLv2
            self.ssl_context.options |= ssl.OP_NO_SSLv3
            self.ssl_context.options |= ssl.OP_NO_TLSv1
            self.ssl_context.options |= ssl.OP_NO_TLSv1_1
            
            self.ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            
            self.logger.info("SSL context initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize SSL context: {str(e)}")
            raise PerplexityAPIError(f"SSL configuration error: {str(e)}")

    async def search_references(
        self,
        query: str,
        model: str = "llama-3.1-sonar-small-128k-online",
        config: Optional[SearchConfig] = None
    ) -> Dict:
        """
        Asynchronously search for academic references with SSL verification disabled
        """
        try:
            config = config or SearchConfig()
            cleaned_query = self._sanitize_query(query)
            payload = self._prepare_payload(cleaned_query, model, config)
            
            # SSL 검증을 비활성화한 커넥터 생성
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                trust_env=True
            ) as session:
                try:
                    return await self._retry_execute_search(session, payload)
                finally:
                    await connector.close()

        except Exception as e:
            self.logger.error(f"Search request failed: {str(e)}")
            raise

    async def _retry_execute_search(
        self,
        session: aiohttp.ClientSession,
        payload: Dict,
        max_retries: int = 3
    ) -> Dict:
        """Execute search with retry mechanism"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Attempt {attempt + 1} of {max_retries}")
                
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    ssl=False  # SSL 검증 비활성화
                ) as response:
                    response_text = await response.text()
                    
                    if response.status != 200:
                        if response.status == 429:  # Rate limit
                            backoff_time = min(30, (2 ** attempt))
                            await asyncio.sleep(backoff_time)
                            continue
                        response.raise_for_status()
                    
                    try:
                        response_json = json.loads(response_text)
                        content = response_json['choices'][0]['message']['content']
                        
                        # JSON 부분 추출
                        json_start = content.find('```json\n') + 8
                        json_end = content.find('\n```', json_start)
                        
                        if json_start == -1 or json_end == -1:
                            raise ValueError("No JSON content found in response")
                            
                        json_content = content[json_start:json_end].strip()
                        parsed_content = json.loads(json_content)
                        
                        return {
                            'references': parsed_content.get('references', []),
                            'metadata': parsed_content.get('metadata', {})
                        }
                        
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse JSON response: {str(e)}")
                        raise
                        
            except aiohttp.ClientResponseError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                break
            
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                break

        raise PerplexityAPIError(f"All retry attempts failed: {str(last_exception)}")


    def _sanitize_query(self, query: str) -> str:
        """Sanitize the search query"""
        if not query.strip():
            raise ValidationError("Search query cannot be empty")
        
        cleaned = re.sub(r'[^\w\s\-.,?!()]', '', query)
        
        if len(cleaned) < 3:
            raise ValidationError("Search query must be at least 3 characters")
        return cleaned

    def _prepare_payload(self, query: str, model: str, config: SearchConfig) -> Dict:
        """Prepare the API request payload"""
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "return_images": config.return_images,
            "return_related_questions": config.return_related_questions,
            "search_recency_filter": config.search_recency_filter,
            "top_k": config.top_k,
            "stream": config.stream,
            "presence_penalty": config.presence_penalty,
            "frequency_penalty": config.frequency_penalty,
            "search_domain": config.search_domain,
        }

    def _process_response(self, response_data: Dict) -> Dict:
        """Process and structure the API response with improved JSON parsing"""
        try:
            self.logger.debug(f"Processing response data: {json.dumps(response_data, indent=2)}")
            
            if not isinstance(response_data, dict):
                raise ValueError(f"Expected dict response, got {type(response_data)}")
                
            if 'choices' not in response_data:
                raise ValueError(f"Missing 'choices' in response. Keys present: {list(response_data.keys())}")
                
            if not response_data['choices']:
                raise ValueError("Empty choices array in response")
                
            choice = response_data['choices'][0]
            if 'message' not in choice:
                raise ValueError(f"Missing 'message' in choice. Keys present: {list(choice.keys())}")
                
            if 'content' not in choice['message']:
                raise ValueError(f"Missing 'content' in message. Keys present: {list(choice['message'].keys())}")
            
            content = choice['message']['content']
            self.logger.debug(f"Content from response: {content}")
            
            # JSON 부분을 추출
            json_start = content.find('```json\n') + 8
            json_end = content.find('\n```', json_start)
            
            if json_start == -1 or json_end == -1:
                self.logger.error("Could not find JSON content in the response")
                raise ValueError("No JSON content found in response")
                
            json_content = content[json_start:json_end].strip()
            self.logger.debug(f"Extracted JSON content: {json_content}")
            
            try:
                parsed_content = json.loads(json_content)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse extracted JSON content: {str(e)}")
                self.logger.error(f"Extracted content was: {json_content}")
                raise
            
            if not isinstance(parsed_content, dict):
                raise ValueError(f"Expected dict after parsing content, got {type(parsed_content)}")
                
            if 'references' not in parsed_content:
                raise ValueError(f"Missing 'references' in parsed content. Keys present: {list(parsed_content.keys())}")
            
            references = parsed_content['references']
            metadata = parsed_content.get('metadata', {})
            metadata['search_timestamp'] = datetime.utcnow().isoformat()
            
            # 참조 검증
            validated_references = []
            for ref in references:
                try:
                    validated_ref = self._validate_reference(ref)
                    validated_references.append(validated_ref)
                except Exception as e:
                    self.logger.warning(f"Failed to validate reference: {str(e)}")
                    self.logger.warning(f"Problematic reference: {ref}")
                    continue
            
            return {
                'references': validated_references,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error processing response: {str(e)}")
            self.logger.error(f"Full response data: {json.dumps(response_data, indent=2)}")
            raise PerplexityAPIError(f"Failed to process API response: {str(e)}") from e

    def _validate_reference(self, reference: Dict) -> Dict:
        """Validate and clean reference data"""
        try:
            expected_fields = set(AcademicReference.__annotations__.keys())
            reference_copy = reference.copy()
            
            # 필수 필드 확인
            required_fields = {'title', 'authors', 'abstract'}
            missing_fields = required_fields - set(reference_copy.keys())
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # authors를 항상 리스트로 변환
            if isinstance(reference_copy.get('authors'), str):
                # 세미콜론이나 쉼표로 구분된 저자 문자열을 리스트로 변환
                authors = reference_copy['authors'].split(';' if ';' in reference_copy['authors'] else ',')
                reference_copy['authors'] = [author.strip() for author in authors]
            
            # citations 처리
            if 'citations' in reference_copy:
                try:
                    reference_copy['citations'] = int(reference_copy['citations'])
                except (ValueError, TypeError):
                    reference_copy['citations'] = None
            else:
                reference_copy['citations'] = None
            
            # 날짜 형식 검증
            if 'publication_date' in reference_copy:
                try:
                    datetime.strptime(reference_copy['publication_date'], '%Y-%m-%d')
                except ValueError:
                    self.logger.warning(f"Invalid date format: {reference_copy['publication_date']}")
                    reference_copy['publication_date'] = None
            
            return reference_copy
            
        except Exception as e:
            self.logger.error(f"Reference validation error: {str(e)}")
            raise ValueError(f"Failed to validate reference: {str(e)}")