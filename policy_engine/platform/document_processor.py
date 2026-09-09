"""Document processing pipeline: PDF → text → LLM objective extraction → policy model update."""

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger

from .database import tenant_session
from .models import PolicyModel, Source
from .storage import download_file

from sqlalchemy import select, update


# ── Generic extraction prompts (not UAE-specific) ────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are analyzing a government or organizational strategy document.
Extract ONLY information explicitly stated in the document. Do not invent or hallucinate data.
Return valid JSON only — no markdown, no commentary."""

OBJECTIVE_EXTRACTION_PROMPT = """Analyze this strategy/policy document and extract structured data as CONCISE JSON.

Document text:
{text}

Return JSON (keep descriptions under 15 words, be brief):
{{
  "document_title": "title",
  "issuing_entity": "who issued it",
  "geography": "country/region",
  "time_horizon": "target year",
  "objectives": [
    {{"id": "OBJ1", "title": "short title", "short": "2-word label", "description": "one short sentence"}}
  ],
  "kpis": [
    {{"id": "KPI1", "objective_id": "OBJ1", "metric": "what is measured", "target": "value", "baseline": "value or null"}}
  ],
  "sectors": ["sector1", "sector2"],
  "stakeholders": ["group1", "group2"]
}}

Rules:
- Extract ALL objectives/goals/pillars. Number OBJ1, OBJ2, etc.
- Extract top 10 quantified KPIs max. Link each to parent objective.
- Keep descriptions SHORT (under 15 words).
- Only extract what is EXPLICITLY in the document.
- Return VALID JSON only. No trailing commas. No markdown."""


async def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Parse PDF bytes into plain text."""
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF required: pip install PyMuPDF")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    full_text = ""
    for page_num in range(page_count):
        page = doc[page_num]
        text = page.get_text()
        full_text += f"\n--- PAGE {page_num + 1} ---\n{text}"
    doc.close()
    logger.info(f"[DocProcessor] Parsed PDF: {page_count} pages, {len(full_text):,} chars")
    return full_text


async def extract_objectives_with_llm(text: str) -> Dict[str, Any]:
    """Use LLM to extract structured objectives from document text."""
    from policy_engine.connectors.kimi import KimiClient

    api_key = os.environ.get("KIMI_API_KEY", "")
    if not api_key:
        logger.warning("[DocProcessor] No KIMI_API_KEY — cannot extract objectives")
        return {}

    client = KimiClient(api_key=api_key)
    prompt = OBJECTIVE_EXTRACTION_PROMPT.format(text=text[:15000])

    try:
        response = await client.chat_completion(
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=8192,
        )
        content = response.get("content", "") or response.get("reasoning", "")

        # Parse JSON from response — handle common LLM output issues
        json_match = content
        if "```json" in content:
            json_match = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_match = content.split("```")[1].split("```")[0]

        json_str = json_match.strip()
        # Fix trailing commas (common LLM issue)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        result = json.loads(json_str)
        logger.info(f"[DocProcessor] Extracted {len(result.get('objectives', []))} objectives, {len(result.get('kpis', []))} KPIs")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"[DocProcessor] Failed to parse LLM JSON: {e}")
        logger.debug(f"[DocProcessor] Raw content (first 500): {content[:500]}")
        return {}
    except Exception as e:
        logger.error(f"[DocProcessor] LLM extraction failed: {e}")
        return {}


async def process_source(tenant_id: str, source_id: str, project_id: str) -> bool:
    """Process a source document: extract text → LLM extraction → update policy model.

    Returns True if successful.
    """
    logger.info(f"[DocProcessor] Processing source {source_id} for project {project_id}")

    try:
        # Load the source record
        async with tenant_session(tenant_id) as db:
            result = await db.execute(
                select(Source).where(Source.id == uuid.UUID(source_id))
            )
            source = result.scalar_one_or_none()
            if not source:
                logger.error(f"[DocProcessor] Source {source_id} not found")
                return False

            # Update status to processing
            source.extraction_status = "processing"
            await db.flush()

        # Get document text
        text = ""
        if source.storage_key:
            # Uploaded file — download from S3
            pdf_bytes = download_file(source.storage_key)
            text = await extract_text_from_pdf(pdf_bytes)
        elif source.url:
            # Web link — download and parse
            import aiohttp
            import ssl
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/pdf,text/html,application/xhtml+xml,*/*",
                "Accept-Language": "en-US,en;q=0.9",
            }
            # Allow self-signed certs (common on government CDNs)
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(
                        source.url,
                        timeout=aiohttp.ClientTimeout(total=120),
                        ssl=ssl_ctx,
                        allow_redirects=True,
                        max_redirects=10,
                    ) as resp:
                        if resp.status == 200:
                            content_type = resp.headers.get("Content-Type", "")
                            data = await resp.read()
                            # Detect PDF by magic bytes, content-type, or URL extension
                            is_pdf = (
                                data[:5] == b"%PDF-"
                                or "pdf" in content_type.lower()
                                or source.url.lower().endswith(".pdf")
                            )
                            if is_pdf:
                                text = await extract_text_from_pdf(data)
                            else:
                                # HTML or text content
                                raw = data.decode("utf-8", errors="replace")
                                # Strip HTML tags for cleaner text extraction
                                import re as _re
                                text = _re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=_re.DOTALL)
                                text = _re.sub(r'<style[^>]*>.*?</style>', '', text, flags=_re.DOTALL)
                                text = _re.sub(r'<[^>]+>', ' ', text)
                                text = _re.sub(r'\s+', ' ', text).strip()[:50000]
                        else:
                            error_msg = f"HTTP {resp.status} from {source.url}"
                            logger.warning(f"[DocProcessor] Failed to fetch: {error_msg}")
                            async with tenant_session(tenant_id) as db:
                                await db.execute(
                                    update(Source).where(Source.id == uuid.UUID(source_id)).values(
                                        extraction_status="failed",
                                    )
                                )
                            return False
            except Exception as fetch_err:
                error_msg = f"Fetch error: {str(fetch_err)[:200]}"
                logger.warning(f"[DocProcessor] {error_msg} for {source.url}")
                async with tenant_session(tenant_id) as db:
                    await db.execute(
                        update(Source).where(Source.id == uuid.UUID(source_id)).values(
                            extraction_status="failed",
                        )
                    )
                return False

        if not text or len(text) < 100:
            async with tenant_session(tenant_id) as db:
                await db.execute(
                    update(Source).where(Source.id == uuid.UUID(source_id)).values(extraction_status="failed")
                )
            logger.warning(f"[DocProcessor] No text extracted from source {source_id}")
            return False

        # Extract objectives via LLM
        extracted = await extract_objectives_with_llm(text)

        if not extracted or not extracted.get("objectives"):
            async with tenant_session(tenant_id) as db:
                await db.execute(
                    update(Source).where(Source.id == uuid.UUID(source_id)).values(extraction_status="completed")
                )
            logger.warning(f"[DocProcessor] No objectives extracted from source {source_id}")
            return False

        # Update or create policy model
        async with tenant_session(tenant_id) as db:
            result = await db.execute(
                select(PolicyModel).where(PolicyModel.project_id == uuid.UUID(project_id))
            )
            pm = result.scalar_one_or_none()

            objectives = extracted.get("objectives", [])
            kpis = extracted.get("kpis", [])
            sectors = extracted.get("sectors", [])

            if pm:
                # Merge with existing — append new objectives, don't overwrite
                existing_ids = {o.get("id") for o in (pm.objectives or [])}
                new_objs = [o for o in objectives if o.get("id") not in existing_ids]
                pm.objectives = (pm.objectives or []) + new_objs
                pm.kpis = (pm.kpis or []) + kpis
                pm.sectors = list(set((pm.sectors or []) + sectors))
                pm.version = (pm.version or 0) + 1
            else:
                pm = PolicyModel(
                    id=uuid.uuid4(),
                    project_id=uuid.UUID(project_id),
                    tenant_id=uuid.UUID(tenant_id),
                    objectives=objectives,
                    kpis=kpis,
                    sectors=sectors,
                    stakeholder_config={"geography": extracted.get("geography", ""), "document_title": extracted.get("document_title", "")},
                )
                db.add(pm)

            await db.flush()

            # Mark source as completed
            await db.execute(
                update(Source).where(Source.id == uuid.UUID(source_id)).values(extraction_status="completed")
            )

        logger.info(f"[DocProcessor] Source {source_id} processed: {len(objectives)} objectives, {len(kpis)} KPIs")
        return True

    except Exception as e:
        logger.error(f"[DocProcessor] Failed to process source {source_id}: {e}")
        async with tenant_session(tenant_id) as db:
            await db.execute(
                update(Source).where(Source.id == uuid.UUID(source_id)).values(extraction_status="failed")
            )
        return False
