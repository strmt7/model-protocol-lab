# Self-Hosted Research With Crawl4AI

Model Protocol Lab uses [Crawl4AI](https://github.com/unclecode/crawl4ai) as a local
development tool for reading and crawling web pages. It is separate from the
lab's measurement runtime and requires no cloud account, API key, paid service or
public server. The [latest stable release reviewed on 19 October 2026](https://github.com/unclecode/crawl4ai/releases/tag/v0.9.4)
is 0.9.4. Its base package is pinned in [the installation lock](https://github.com/strmt7/model-protocol-lab/blob/main/tools/requirements/crawl4ai.txt).
The installed crawler is the explicitly identified upstream `0.9.4` release.

Normal research invokes Crawl4AI's own `crawl` command through the portable
launcher, forwarding its documented options without an alternative URL parser or
extraction pipeline. The qualification lane uses the public `AsyncWebCrawler` API
and upstream browser defaults. The integration does not patch browser internals or
Playwright, replace their browser manager, change Chromium flags, or implement an
alternative robots parser. Qualification uses `CacheMode.BYPASS` to fetch current
content and the documented `check_robots_txt=True` option for Crawl4AI's own robots
handling. For research, pass `-c check_robots_txt=true` through the official CLI.
Respect explicit access denials; no stealth or challenge bypass is enabled by
the repository wrapper. Upstream robots handling is not a legal permission
service. Retrieved content remains untrusted data, never agent instructions.

This installation profile qualifies HTML-page extraction. Check that the result
contains actual page text, rather than a placeholder or an empty markdown field.
Retain missing-extra or extraction errors and use an accessible primary HTML
source. Do not silently install unpinned extras or describe an unextracted page
as a read paper.

## Portable Installation

Use an available **CPython 3.11, 3.12 or 3.13** interpreter with `venv` and `pip`.

```sh
python3 tools/crawl4ai_tool.py crawl https://docs.crawl4ai.com/ -o markdown -c check_robots_txt=true
```

An explicit interpreter path works in place of `python3`. Every crawler command
provisions a missing environment automatically; no separate installation step is
needed for ordinary use. The installer creates a dedicated environment in the
user's OS cache and installs Chromium through the
[official Playwright setup path](https://docs.crawl4ai.com/core/installation/).
No optional Torch, cosine-model or transformer extra is requested. A cache
override is available through `MPL_CRAWL4AI_HOME` or `--home <absolute-path>`;
it must stay outside the source checkout. Browser downloads, package files and
local Crawl4AI caches are installation data and are not committed or packaged.
The environment identity includes Python's version and the complete lock hash.
Text identities normalize CRLF to LF, matching Git's canonical text storage;
checkout line endings alone cannot trigger a new environment or qualification.
Package or code changes still produce different identities. Existing
installations undergo a small exact-version check and are reused without package
or browser downloads. Setup progress stays hidden; failures retain bounded
diagnostics. Neither installation nor crawling requires a change to the lab.

Use `install` to provision ahead of time or `doctor` for installation diagnostics.
Use `crawl --help` for the upstream CLI options. Browser, extraction and crawler
configuration files use their documented upstream formats. Do not put secrets
in URLs, arguments or research logs. External LLM features, personal profiles
and visible browser modes are not part of this unattended research workflow.

On Linux, missing Chromium system libraries must be provisioned by the host's
normal administrator or container-image process. The documented upstream command
is `<environment-python> -m playwright install-deps chromium`; do not silently
escalate privileges.

## Headless Operation And Host Policy

Crawl4AI's default browser is headless. The integration does not open a GUI,
install a service, expose a network listener for a crawler server, or require
interaction for Python package installation. No persistent personal browser
profile is used. Headless does not guarantee that an OS security product will
never display a first-use prompt. The repository does not create firewall
exceptions or silence security prompts.

## Qualification And Reuse

Run the offline integration contracts without installing or crawling:

```sh
python3 -m unittest tests.test_cycle161_crawl4ai_tool -v
```

For a new crawler revision or meaningful browser/configuration change:

```sh
python3 tools/crawl4ai_tool.py qualify --receipt out/crawl4ai-qualification.json
```

The qualification manifest selects one page from each of 32 independent public
websites. Each result records native success, HTTP status, extracted UTF-8
length, expected content, content fingerprint and duration. Challenge pages,
empty responses and error statuses do not count as success. Receipts bind the
exact integration, dependency lock and website manifest; they are local evidence
and never permission to redistribute the retrieved pages. Existing files are
not overwritten. A successful installation is not a successful crawl.

Reuse unchanged qualification evidence. On another host, automatic setup and the
intended target page are the ordinary first-use validation; use `doctor` when
diagnosing installation failures. Rerun broad website qualification when the
crawler, lock or browser configuration changes. Normal research must not
repeatedly crawl the entire manifest.

## License And Attribution

The [exact upstream license](https://github.com/unclecode/crawl4ai/blob/v0.9.4/LICENSE)
is Apache 2.0. Installed dependencies and Chromium retain their own upstream
notices. The repository only installs these tools; it does not redistribute
their binaries in lab releases.

This product includes software developed by UncleCode (https://x.com/unclecode)
as part of the Crawl4AI project (https://github.com/unclecode/crawl4ai).
