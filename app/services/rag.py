"""RAG (retrieval-augmented generation) service: chunking, embedding, and access-scoped search.

Access control is enforced entirely in this module using group IDs resolved
server-side from the authenticated user (never from LLM- or request-supplied
group/user IDs): ``search`` only ever queries documents whose ``group_id`` is
one the caller belongs to, and ``ingest_document``/``delete_document`` require
the caller to be an "admin" member of the target group.
"""

