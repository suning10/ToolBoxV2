import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        _FRONTMATTER_DELIMITER = "---"
        raw = """
        ---
name: web_research
description: Use when the user asks something needing current facts, news, prices, or other information you shouldn't answer from memory alone.
---

# Web Research

Use the `duckduckgo_search_tool` to find current information rather than answering from memory when the question is about recent events, prices, availability, or anything else time-sensitive.

## Steps

1. Break the user's question into one or more concise search queries — don't dump the whole question verbatim into the search tool.
2. Call `duckduckgo_search_tool` with your first query.
3. If the results are thin or ambiguous, refine the query and search again rather than guessing.
4. Cross-check facts across at least two independent results before stating something as fact.
5. When you answer, briefly note where the information came from (e.g. "according to X").
6. If results are stale, conflicting, or insufficient, say so explicitly rather than fabricating a confident answer.

## When to stop

- Once you have enough corroborating information to answer confidently, stop searching and respond.
- Don't run more than 2-3 searches for a single question — if it's still unresolved after that, tell the user what you found and what remains unclear.

        """
        parts = raw.split(_FRONTMATTER_DELIMITER, 2)
        print(parts)
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
