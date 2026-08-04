"""
prompt_builder.py

Purpose (SDD Section 14 -- Prompt Template):
- Take the user's question + the retrieved chunks from retriever.py.
- Build a single prompt that:
  1. Gives Ollama ONLY the retrieved legal text as its source of truth
     (never lets it answer from its own parametric memory).
  2. Forces a structured 5-field output: Applicable Law, Explanation,
     Citizen Rights, Penalty, Source.
  3. Explicitly instructs it to say so if the retrieved context doesn't
     actually answer the question.
  4. LABELS sources by rank (PRIMARY SOURCE vs ADDITIONAL CONTEXT) and
     instructs the model to prefer the primary one -- this fixes a real
     bug we found in testing: when handed multiple relevant-looking
     chunks with no ranking signal, Ollama sometimes cited a secondary
     source over the actual top-ranked match (e.g. picking Article 351
     over the Eighth Schedule for an "official languages" question,
     even though the Schedule was the highest-scoring retrieved chunk).
     Retrieval was correct; the model's unranked choice wasn't.

Keeping the template as a separate function (not inline in rag_engine.py)
means you can iterate on prompt wording without touching orchestration logic.
"""

SYSTEM_INSTRUCTIONS = """You are a legal assistant for Indian citizens. You must answer ONLY using the CONTEXT provided below, which contains real excerpts from the Constitution of India, ranked from most to least relevant. Do not use any outside knowledge, even if you know it.

STRICT RULE ON SOURCE SELECTION: The context is pre-ranked by a search system that has already determined the [PRIMARY SOURCE] is the single best match for this question. You MUST use the [PRIMARY SOURCE] for your "Applicable Law" and "Source" fields. Do NOT switch to an [ADDITIONAL CONTEXT] source just because it seems more detailed, more specific, or easier to explain -- the ranking has already accounted for relevance. The ONLY exception: if [PRIMARY SOURCE] is completely unrelated to the question's topic (not just less detailed), you may use an [ADDITIONAL CONTEXT] source instead, and you must explicitly say why in your Explanation.

Respond in EXACTLY this format:

Applicable Law: <the specific Article/Schedule number and its short title>
Explanation: <plain-language explanation of what this means, based only on the context>
Citizen Rights: <what this means for the citizen's rights, based only on the context>
Penalty: <any penalty/consequence mentioned in the context, or "Not specified in this provision" if none>
Source: <exact Article/Schedule number(s) used>

If the CONTEXT does not contain enough information to answer the question, respond with EXACTLY:
"I don't have enough information in the Constitution to answer this confidently. This may be covered under a different law not yet in my database."

Do not invent article numbers. Do not answer from memory. Use only the CONTEXT below."""


def format_context(chunks: list[dict]) -> str:
    """
    Turn retrieved chunks into a labeled, RANKED context block.
    Chunks are assumed to already be sorted best-to-worst (retriever.py's
    RRF fusion does this) -- we just add explicit rank labels here so the
    LLM doesn't have to guess which source is strongest.
    """
    blocks = []
    for i, c in enumerate(chunks):
        rank_label = "PRIMARY SOURCE" if i == 0 else f"ADDITIONAL CONTEXT #{i}"

        if c.get("article_number"):
            label = f"Article {c['article_number']}"
            if c.get("title"):
                label += f" — {c['title']}"
        else:
            label = c.get("schedule_name", "Unknown Source")
            if c.get("part"):
                label += f" ({c['part']})"

        blocks.append(f"[{rank_label}: {label}]\n{c['text']}")
    return "\n\n".join(blocks)


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = format_context(chunks)
    return f"""{SYSTEM_INSTRUCTIONS}

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


if __name__ == "__main__":
    # Manual test with fake chunks -- verifies formatting without needing
    # the retriever or Ollama running. Includes 2 chunks to confirm the
    # PRIMARY SOURCE / ADDITIONAL CONTEXT labeling renders correctly.
    fake_chunks = [
        {
            "article_number": "21",
            "title": "Protection of life and personal liberty",
            "text": "No person shall be deprived of his life or personal liberty except according to procedure established by law.",
        },
        {
            "article_number": None,
            "schedule_name": "EIGHTH Schedule",
            "part": None,
            "title": "Eighth Schedule — Official Languages",
            "text": "Assamese, Bengali, Gujarati, Hindi, Kannada, Kashmiri...",
        },
    ]
    prompt = build_prompt("What is my right to life?", fake_chunks)
    print(prompt)