---
name: skill-creator
description: Create new skills, modify and improve existing skills, and measure skill performance. Use when users want to create a skill from scratch, edit, or optimize an existing skill, run evals to test a skill, benchmark skill performance with variance analysis, or optimize a skill's description for better triggering accuracy.
---

## Purpose
Search the database for records matching a keyword.

## When To Use
- User asks for specific records or data
- Need to look up information by keyword
- User says "find", "search", "look up", "get"

## When NOT To Use
- Do not use for writing or updating data
- Do not use if user asks for analytics → use analytics_skill instead

## Input
| Parameter | Type | Required | Description              |
|-----------|------|----------|--------------------------|
| query     | str  | ✅       | keyword or phrase        |
| limit     | int  | ❌       | max records (default 10) |

## Output
- On success: list of matching records as JSON string
- On failure: error message with reason

## Routes To
- `process_results` → on success
- `handle_error`    → on failure, LLM will retry with simpler query

## Example
User: "show me orders from John"
→ search_db(query="John orders")

## Notes
- Keep queries short (1-3 keywords works best)
- Retry with broader terms if no results