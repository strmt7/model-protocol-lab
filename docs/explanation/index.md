# Explanation

Explanation is where the reasoning lives: why the lab measures the
prompt-format confound the way it does, and the design record behind
each instrument.

## The idea

A checkpoint's behaviour is only ever observed through a rendered
conversation. Change the rendering — BOS handling, turn markers,
prefixes, whitespace, field order — and the token sequence changes,
so the behaviour can change with the weights held fixed. The lab
exploits that: it renders the same fixtures under many format
hypotheses as exact token sequences and compares answers and response
boundaries token-for-token. A failure that disappears under a
different rendering is proven prompt-fixable; the lab can never prove
a weights limitation, only fail to find a fixing format in the space
it searches.

Three design decisions shape every measurement:

- **Boundaries are token-ID subsequences**, not string stop
  sequences — one string can be many token sequences, and the two
  notions diverging is itself a diagnostic.
- **Generation is exactly N tokens with no implicit EOS stop**, so
  stopping behaviour is observed rather than assumed.
- **Selection abstains on ties and failures** — a conservative
  single-winner rule over a strict search/held-out split, so a
  reported winner has survived evidence it was not selected on.

## The design record

Every probe in USAGE §8 has a design document: the question, the
population, the scorer, the decision rule, and the adversarial review
that shaped it — including the rounds that returned NO-BUILD, whose
documents record the executed counterexamples that killed the sketch
and the narrower resubmission candidates they left behind. The full
list is in this section's navigation; the one-line index of the
decisions themselves is the
[architecture decision index](adr-index.md). The chronological record of
the whole programme is the [research log](../RESEARCH_LOG.md);
release readiness is assessed in
[RELEASE_READINESS_0.1.0.md](../RELEASE_READINESS_0.1.0.md), and the
portability study behind the doctor matrix is
[RESEARCH_PORTABILITY_2026-10-15.md](../RESEARCH_PORTABILITY_2026-10-15.md).
