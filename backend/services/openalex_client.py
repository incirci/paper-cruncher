"""OpenAlex API Client."""

import asyncio
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from backend.core.config import settings


class OpenAlexClient:
    """Client for interacting with OpenAlex API."""

    BASE_URL = "https://api.openalex.org"

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Ensure cache directory exists
        self.cache_dir = settings.get_vector_db_path().parent / "cache" / "citations"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def _make_request(self, client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> httpx.Response:
        """Make a request with retry logic."""
        retries = 3
        base_delay = 1.0
        
        for i in range(retries):
            try:
                response = await client.get(url, params=params)
                if response.status_code == 429:
                    if i == retries - 1:
                        response.raise_for_status()
                    
                    wait_time = base_delay * (2 ** i)
                    self.logger.warning(f"Rate limited (429) by OpenAlex. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                     if i == retries - 1:
                        raise e
                     wait_time = base_delay * (2 ** i)
                     self.logger.warning(f"Rate limited (429) by OpenAlex. Retrying in {wait_time}s...")
                     await asyncio.sleep(wait_time)
                     continue
                raise e
            except Exception as e:
                # Handle connection errors etc
                if i == retries - 1:
                    raise e
                wait_time = base_delay * (2 ** i)
                self.logger.warning(f"Request failed ({e}). Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        raise Exception("Max retries exceeded")

    async def search_paper(self, title: str) -> Optional[str]:
        """Search for a paper by title and return its OpenAlex ID."""
        self.logger.info(f"Searching OpenAlex for: {title}")
        async with httpx.AsyncClient() as client:
            params = {
                "filter": f"title.search:{title}",
                "per-page": 1
            }
            try:
                response = await self._make_request(client, f"{self.BASE_URL}/works", params=params)
                data = response.json()
                
                results = data.get("results", [])
                if results:
                    return results[0]["id"]
            except Exception as e:
                self.logger.error(f"Search failed: {e}")
            return None

    async def fetch_basic_metadata(self, work_id: str) -> Dict[str, Any]:
        """Fetch only basic metadata (no references/citations) for indexing."""
        self.logger.info(f"Fetching basic metadata for {work_id}")
        async with httpx.AsyncClient() as client:
            response = await self._make_request(client, f"{self.BASE_URL}/works/{work_id}", params={})
            work = response.json()
            
            # Normalize just the basics
            authors = work.get("authorships", [])
            topics = work.get("topics", [])
            primary_topic = topics[0]["display_name"] if topics else "Unknown Topic"
            
            return {
                "title": work.get("title"),
                "year": work.get("publication_year"),
                "authors": [{"name": a["author"]["display_name"]} for a in authors],
                "primary_topic": primary_topic,
                "citation_count": work.get("cited_by_count", 0),
                "url": work.get("doi") or work.get("id")
            }

    async def fetch_details(self, work_id: str) -> Dict[str, Any]:
        """Fetch references and citations for a paper from OpenAlex."""
        short_id = work_id.split("/")[-1]
        cache_path = self.cache_dir / f"oa_{short_id}.json"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    self.logger.info(f"Loading {short_id} from cache")
                    return cached_data
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {short_id}: {e}")

        self.logger.info(f"Fetching details for {short_id} from OpenAlex")
        async with httpx.AsyncClient() as client:
            # 1. Fetch Work Details
            response = await self._make_request(client, f"{self.BASE_URL}/works/{work_id}", params={})
            work = response.json()
            
            # 2. Fetch References
            ref_ids = work.get("referenced_works", [])
            references = []
            
            # Batch fetch references details
            chunk_size = 50
            for i in range(0, len(ref_ids), chunk_size):
                chunk = ref_ids[i:i + chunk_size]
                # chunk contains full URLs, we need just the IDs (W...)
                chunk_short_ids = [url.split("/")[-1] for url in chunk]
                ids_str = "|".join(chunk_short_ids)
                
                try:
                    ref_resp = await self._make_request(
                        client,
                        f"{self.BASE_URL}/works", 
                        params={"filter": f"openalex_id:{ids_str}", "per-page": chunk_size}
                    )
                    ref_data = ref_resp.json().get("results", [])
                    references.extend(ref_data)
                except Exception as e:
                    self.logger.error(f"OpenAlex ref batch failed: {e}")
                await asyncio.sleep(0.1)

            # 3. Fetch Citations (Works that reference this work)
            citations = []
            try:
                cite_resp = await self._make_request(
                    client,
                    f"{self.BASE_URL}/works",
                    params={"filter": f"referenced_works:{work_id}", "per-page": 200}
                )
                citations = cite_resp.json().get("results", [])
            except Exception as e:
                self.logger.error(f"OpenAlex citations fetch failed: {e}")

            # 4. Normalize Data
            def normalize_work(w):
                authors = w.get("authorships", [])
                topics = w.get("topics", [])
                primary_topic = topics[0]["display_name"] if topics else "Unknown Topic"
                concepts = [c["display_name"] for c in w.get("concepts", [])[:3]]
                
                return {
                    "title": w.get("title"),
                    "year": w.get("publication_year"),
                    "authors": [{"name": a["author"]["display_name"]} for a in authors],
                    "primary_topic": primary_topic,
                    "concepts": concepts,
                    "citation_count": w.get("cited_by_count", 0),
                    "url": w.get("doi") or w.get("id")
                }

            data = {
                "title": work.get("title"),
                "year": work.get("publication_year"),
                "authors": [{"name": a["author"]["display_name"]} for a in work.get("authorships", [])],
                "primary_topic": work.get("topics", [])[0]["display_name"] if work.get("topics") else "Unknown Topic",
                "concepts": [c["display_name"] for c in work.get("concepts", [])[:3]],
                "references": [normalize_work(r) for r in references],
                "citations": [normalize_work(c) for c in citations],
                "citation_count": work.get("cited_by_count", 0),
                "url": work.get("doi") or work.get("id")
            }
            
            # Save to cache
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
            except Exception as e:
                self.logger.warning(f"Failed to save cache for {work_id}: {e}")
                
            return data
