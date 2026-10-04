import asyncio
import sys
import httpx
sys.path.insert(0, ".")
from backend.config import settings

async def check_gan_paper():
    # Fetch paper details for Goodfellow 2014 from S2
    headers = {"User-Agent": "ResearchLens/2.0"}
    if settings.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY.strip()
    async with httpx.AsyncClient(timeout=20.0) as client:
        # Search for GENERATIVE ADVERSARIAL NETS
        resp = await client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": "GENERATIVE ADVERSARIAL NETS",
                "limit": 5,
                "fields": "paperId,title,externalIds,openAccessPdf,publicationVenue,year"
            },
            headers=headers
        )
        print("S2 Search status:", resp.status_code)
        if resp.status_code == 200:
            for p in resp.json().get("data", []):
                print(f"Paper: {p.get('title')} ({p.get('year')})")
                print(f"  paperId: {p.get('paperId')}")
                print(f"  externalIds: {p.get('externalIds')}")
                print(f"  openAccessPdf: {p.get('openAccessPdf')}")
                print(f"  doi: {p.get('doi')}")

asyncio.run(check_gan_paper())
