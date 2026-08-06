"""
web_search.py

Purpose:
- Fetch real-time web information about legal procedures, FIR filing,
  complaint mechanisms, government portals, etc.
- This runs IN PARALLEL with the RAG pipeline (doesn't block Llama)
- Results are appended AFTER the trusted Constitution-based answer

Why separate from RAG:
- The Constitution text (from Llama + RAG) is 100% verified from official sources
- Web search provides PRACTICAL guidance (how-to steps, forms, links)
- Keeping them separate ensures users can distinguish between:
  1. Constitutional rights (verified, from dataset)
  2. Practical procedures (from web, may need verification)
"""

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from typing import Optional


DDG_MAX_RESULTS = 8  # Number of web results to fetch


class WebSearchClient:
    """
    Lightweight wrapper around DuckDuckGo Search API.
    No API key needed -- completely free.
    """
    
    def __init__(self, max_results: int = DDG_MAX_RESULTS):
        self.max_results = max_results
    
    def search(self, query: str) -> list[dict]:
        """
        Search the web for practical legal guidance.
        
        Returns list of results with:
        - title: Result title
        - url: Link to the source
        - snippet: Short description/excerpt
        - source: Domain name (for credibility check)
        """
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
            
            formatted_results = []
            for r in results:
                # Filter out irrelevant results (file sharing, generic Google pages)
                if self._is_irrelevant_result(r):
                    continue
                    
                formatted_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "source": self._extract_domain(r.get("href", ""))
                })
            
            return formatted_results[:5]  # Return top 5 relevant results
        
        except Exception as e:
            print(f"[web_search] Error during DuckDuckGo search: {e}")
            return []
    
    def _is_irrelevant_result(self, result: dict) -> bool:
        """Filter out obviously irrelevant results like file sharing sites."""
        title = result.get("title", "").lower()
        url = result.get("href", "").lower()
        body = result.get("body", "").lower()
        
        # Keywords that indicate irrelevant results
        irrelevant_keywords = [
            "google drive", "file sharing", "free files",
            "files.google.com", "drive.google.com",
            "file.io", "super simple file"
        ]
        
        text_to_check = f"{title} {url} {body}"
        return any(keyword in text_to_check for keyword in irrelevant_keywords)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL for source credibility."""
        if not url:
            return "Unknown"
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "")
        except:
            return "Unknown"
    
    def is_available(self) -> bool:
        """Quick health check -- can we reach DuckDuckGo?"""
        try:
            test_results = self.search("legal aid India")
            return len(test_results) > 0
        except:
            return False


def build_practical_suggestions_query(constitutional_topic: str, user_query: str = "") -> str:
    """
    Convert a constitutional topic into a practical web search query.
    
    Examples:
    - "Article 21 right to life" → "how to file FIR India police complaint procedure"
    - "Article 19 freedom of speech" → "freedom of speech India legal protection how to exercise"
    - "Article 21A right to education" → "RTI education India how to apply procedure"
    - User query "How to file FIR for theft?" → "how to file FIR for theft India online procedure"
    """
    # If the user query itself is already a practical question (contains action words), use it directly
    if user_query:
        user_query_lower = user_query.lower()
        action_keywords = ["how to", "file", "proceed", "complaint", "fir", "report", 
                          "apply", "register", "step", "procedure", "process"]
        if any(kw in user_query_lower for kw in action_keywords):
            # Enhance with India-specific context if not already present
            if "india" not in user_query_lower and "indian" not in user_query_lower:
                return f"{user_query} India procedure"
            return user_query
    
    # Map constitutional topics to practical search queries
    query_mappings = {
        "right to life": "how to file FIR criminal complaint India police procedure",
        "personal liberty": "illegal detention habeas corpus India how to file petition",
        "freedom of speech": "freedom of speech expression India legal rights how to exercise",
        "right to education": "Right to Education Act India school admission complaint procedure",
        "equality": "discrimination complaint India human rights commission procedure",
        "constitutional remedies": "how to file writ petition High Court Supreme Court India",
        "life liberty": "police complaint procedure India online FIR filing guide",
        "fundamental rights": "violation of fundamental rights India where to complain NHRC",
        "theft": "how to file FIR theft India online police complaint procedure",
        "fir": "how to file FIR India online procedure police complaint step by step",
        "complaint": "how to file police complaint India procedure online offline",
        "returns": "cooperative society returns filing India procedure form deadline",
    }
    
    # Check if any mapped topic matches
    constitutional_topic_lower = constitutional_topic.lower()
    for topic_keyword, practical_query in query_mappings.items():
        if topic_keyword in constitutional_topic_lower:
            return practical_query
    
    # Default: generate a generic practical query focused on procedures
    return f"{constitutional_topic} India legal procedure how to file complaint application steps"


if __name__ == "__main__":
    # Manual test
    client = WebSearchClient()
    
    print("Testing DuckDuckGo search availability...")
    if not client.is_available():
        print("WARNING: DuckDuckGo search may be unavailable.")
    else:
        print("DuckDuckGo search is available.\n")
    
    test_queries = [
        "how to file FIR India online procedure",
        "Right to Education Act India complaint procedure",
        "habeas corpus petition India how to file",
    ]
    
    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {q}")
        print('='*70)
        results = client.search(q)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['title']}")
            print(f"   Source: {r['source']}")
            print(f"   URL: {r['url']}")
            print(f"   Snippet: {r['snippet'][:150]}...")
