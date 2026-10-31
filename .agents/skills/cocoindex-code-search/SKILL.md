---
name: cocoindex-code-search
description: Route broad conceptual code searches through the pinned local index, then verify exact source.
origin: adapted from strmt7/FlitzZip .agents/skills/cocoindex-code-search/SKILL.md (itself adapted from strmt7/VulnerabilityScreener)
---

# CocoIndex Code Search

This workflow is mandatory for broad or fuzzy repository code navigation, before
opening many files. Keep direct `rg` first for exact symbols, strings,
counts and small known scopes. CocoIndex is a development aid and never
part of the lab's measurement runtime or training data.

1. Use the `AGENTS.md` reading map to identify the likely domain. If the local
   index is missing, run `<python> tools/cocoindex_agent_search.py install`,
   then `<python> tools/cocoindex_agent_search.py index`.
   The first index may download a local embedding
   model and take several minutes. Installation and index data stay in a
   computed user cache, outside the live checkout; `AGENT_CODE_HOME` may
   override it with an absolute path.
2. Run `<python> tools/cocoindex_agent_search.py search --limit 5
   "conceptual question"`. Search rejects source changes until `index`
   refreshes the code-only mirror. Successful searches record a local,
   per-checkout receipt with the source/config digests and query hash; an
   installation or index marker alone does not prove search use. This receipt
   is development evidence, not a proof of complete source review. The wrapper disables usage telemetry
   and uses CPU embeddings.
3. Confirm each relevant candidate with `rg -n` and an exact source
   read. The first-five ranking is a hint, not evidence of completeness
   or correctness. If semantic results miss the target, use bounded
   `rg -l` and refine once. Never change code based on a semantic hit
   alone.
4. Exact `--path` file filters work. CocoIndex Code 0.2.42 returned
   empty results for wildcard filters on the reference host;
   this wrapper rejects them. Use unscoped semantic routing followed by
   `rg` for path narrowing.
5. If installation, indexing or search fails, retain and disclose the exact
   error, then restore the tool before continuing broad conceptual navigation.
   Do not substitute `rg` for the required semantic search. Continue independent
   work and exact reads of already known files or symbols while restoring it.

Use `tools/agent_context.py startup` at a new context boundary to load both
mandatory skills. The bounded document reader supplements code navigation;
it does not replace semantic routing or the authoritative source reads.

Do not register over an existing `cocoindex-code` MCP server bound to
another repository. This repository's CLI workflow is independent of
that host configuration and transfers through tracked files.
