# Compatibility-form substitution — design (cycle 231)

`run --compatibility-form`: substitute one compatibility-form
character (a ligature, an enclosed digit, a vulgar fraction, a
superscript digit, a Roman numeral, or a halfwidth katakana
letter) for its plain-text reading in one field of a carrier
marker sentence inside a generated fact-recall panel, and
measure — per candidate (template family) and per class,
behind an encode-time token-id gate and a completion guard —
whether the form alone moves the model's answers to wrong
in-grammar values.

This document is the build contract. It consolidates the
wave-12 round-3 candidate with the fresh review's amendments
A1–A9 folded in (adjudication: BUILD-as-amended, coordinator
#80, 2026-10-31; built in cycle 231). Where an amendment
changed the candidate's text, the amended text is what
appears below; the amendment list follows so a reader can
see what moved.

## Amendments folded in (A1–A9)

- **A1 (cap derivation restated; OC conditioned on the
  divergent-domain size M).** The §6 cap bounds incomplete
  cases *within the domain* at 6; it guarantees paired ≥ 18
  only when the domain is the full 24 cases. When the ids
  gate shrinks the domain to M, labels issue at paired
  counts down to M − 6 (and to M at k = 0), with operating
  characteristics conditioned on M; §6 prints the
  M-conditioned rows, every report prints the arm's M
  beside its label, and §7's acquittal caveat is
  conditioned on M. The decision rule, the cap value, and
  the floors are unchanged.
- **A2 (latent-path pins).** The screen-failure detail is
  `compatibility_form_screen_failed` everywhere (the
  candidate §3's `compat_fold_screen_failed` spelling is
  corrected to it; the draw-material namespace `compatfold`
  is a separate internal identifier and is unchanged).
  §3's give-up sentence pins the counting exactly as the
  module implements it: the redraw counter increments once
  per colliding redraw, and exhaustion is raised when the
  counter exceeds 1000.
- **A3 (control arm's decomposition table pinned).** §5
  states that the control arm's class-4 table is the empty
  set: class 4 (`echo_compat`) never fires for control
  cells, and a folded control answer classifies
  `form_only`. The battery carries the witness (a
  fullwidth rendering of the expected value under the
  control arm → `form_only`).
- **A4 (staged counts carry their universe and
  operator).** §2's staged counts state their universe:
  the ligature staging ranges by name *prefix* over
  decomposition-bearing rows (8; file-wide by prefix 9,
  the extra row U+0153 carrying no decomposition and
  being NFKC-stable); the Roman staging ranges by name
  *substring* over decomposition-bearing rows (32 = 16
  uppercase + 16 small; file-wide 24 by prefix, 40 by
  substring, the extra rows likewise decomposition-free
  or NFKC-stable).
- **A5 (§5 sentence-level truthfulness).** (i) The
  classes-4+5 gloss is restated in the predicate's own
  terms: together they are exactly the answers whose NFKC
  fold is the expected value, split by whether the
  treated class supplied the forms; case-changing
  renderings (circled capitals, squared letters) fold to
  a different string and decompose `non_answer`, never
  toward J. (ii) The battery enumeration lists all 15
  executed cases, including the halfwidth-kana
  `non_answer` case. (iii) The echo-reachability sentence
  states the executed record (design-phase sample of two
  numbers; exhaustive confirmation over all 216,000
  expected values executed in the review round by both
  red teams), and the build's decomposition battery
  includes the exhaustive reachability search as a test.
- **A6 (considered exclusions disclosed).** §2 names the
  excluded mirror families with reasons: the `<sub>`
  subscript digits U+2080–2089 (the class list is
  superscripts; a subscript field would be a seventh
  class and a population change, not an amendment),
  parenthesized digits (9 rows; multi-character readings
  break the one-character field uniformity), and circled
  numbers (41 rows; two-digit readings, same reason). The
  population does not move.
- **A7 (arithmetic screen stated exactly).** §3's screen
  sentence names its operator set (`+`, `*`, and the six
  arithmetic words) and states the guarantee in its true
  form: no arithmetic is requested — the question is
  byte-identical to the frozen template and references no
  marker field; the fraction reading's solidus is a
  transcription character inside an unreferenced field,
  disclosed, not an operator in any task.
- **A8 (§7 gloss carries the third-form clause).** §7's
  falsification gloss carries: for the fraction arm under
  an NFKC-class normalizer, the form that reached the
  model is the pipeline's rewritten `1⁄2` form (§6), and
  a fire is attributable to that rewritten form against
  the ASCII reading.
- **A9 (§11 sweep completed).** §11's opening is scoped
  to classified instruments over non-`<wide>`
  populations, and the neighbour list gains §8.31
  template surface-noise robustness (`surfacenoise.py`,
  `unicode_variant` operator) with its disposition.

One census correction from the build (reality wins over
the design record): the candidate's naming audit counted
82 distinct long options on the `run` parser; the tree
had gained one more by build time, so the build-time
census is 83, and `--compatibility-form` — verified free
in both counts — becomes the 84th.

## 1. Monday-morning utility

Anyone pasting typeset text (PDF extraction is full of
ligatures, enclosed digits, and vulgar fractions) into a
local model learns, Monday morning, whether the paste
silently changes answers — and platform teams learn, per
candidate (template family) and per compatibility class,
whether an NFKC normalization pass at ingest is
protective, neutral, or harmful for that family.

**The prior, stated plainly: no direct LLM prior exists
for this probe.** A dedicated literature sweep in the
wave-12 research found no published study of LLM
behaviour on NFKC-foldable characters. The justification
is therefore three non-LLM legs, and the design leans on
nothing else: (a) *tokenizer mechanics* — whether a form
survives to the model at all is decided by the
tokenizer's normalizer, which the §6 ids gate measures
per family; (b) *the standard's own warning* — UAX #15
§1.2 (rev. 58): the K forms "must not be blindly applied
to arbitrary text. Because they erase many formatting
distinctions … they may remove distinctions that are
important to the semantics of the text"; compatibility
equivalence (§1.1) is the *weaker* equivalence whose
distinctions "may represent a visual distinction that is
significant in some textual contexts"; (c) *the paste
path* — ligatures, enclosed alphanumerics, vulgar
fractions, superscripts, Roman numerals, and halfwidth
katakana are exactly the characters typeset and
PDF-extracted text carries into prompts. Boucher et al.
(arXiv:2106.09898) is supporting only — encoding-level
perturbations change deployed NLP behaviour — and is
never load-bearing here. **A null result is a legitimate
outcome frozen in advance** (§7 says so in advance; §9 R1
forbids dose escalation in response to one).

Framing: this is canonical-form's (§8.54) missing third
leg — the sibling instrument measures *canonical*
equivalence (NFC/NFD); this probe measures
*compatibility* equivalence (the forms NFKC folds and
NFC/NFD keep distinct), with the identical instrument
shape: generated fact-recall panel, ASCII expected
values, encode-time token-id gating, digest-committed
records, paired classification with floors derived under
this probe's own null. Uniqueness is claimed for the
combination only, never "first" (§11).

## 2. The population definition

**Version pin.** The population is derived from
**`UnicodeData.txt` Version 15.0.0**
(`https://www.unicode.org/Public/15.0.0/ucd/UnicodeData.txt`,
SHA-256
`806e9aed65037197f1ec85e12be6e8cd870fc5608b4de0fffd990f689f376a73`;
34,924 lines). Reasons, in order of weight: (1)
*single-version coherence* — the interpreter that runs
the build has `unicodedata.unidata_version = 15.0.0`
(Python 3.12.3), and every population check is run
against both the pinned file and the local interpreter,
which must agree; (2) *round-2 precedent* — the
confusable-skeleton population pinned its data to 15.0.0
for the same reason; (3) the *cited text* (UAX #15
rev. 58 / Unicode 18.0.0, §14) and the *data* are
different artefacts — the annex is cited for definitions
and its worked examples; the data the definitions
consume is pinned separately, and compatibility mappings
are stable for the frozen characters below
(normalization stability), so no arm, floor, or claim
depends on post-15.0 additions. A future Unicode bump
is a population change requiring re-derivation, not a
silent upgrade.

**General eligibility (frozen, all classes).** A
character `ch` is eligible iff **all** of:

- its decomposition field in the pinned file is a
  **tagged** compatibility mapping (first token `<...>` —
  e.g. `<compat>`, `<circle>`, `<fraction>`, `<super>`,
  `<narrow>`);
- `NFC(ch) == NFD(ch) == ch` (no canonical decomposition
  — the canonical-form boundary: this population is
  invisible to §8.54's instrument by construction);
- `NFKC(ch) != ch` (it actually folds);
- its tag is **not `<wide>`**.

**The `<wide>` exclusion, executed.** Fullwidth forms
are perturbation's `unicode_variant` territory and are
excluded mechanically by the last clause: the pinned
file carries **104** `<wide>` rows, **0** of them in any
class below. Census context, executed: 5,857 file rows
carry a decomposition; 3,796 are tagged; 3,692 satisfy
the general rule (3,796 − 104 `<wide>`).

**Class rules (each is the general rule plus one
clause; staged counts executed so a reviewer sees what
each clause removes; per A4, staged name counts state
their universe and operator):**

| class | additional clause | staged counts |
|---|---|---|
| ligatures | name starts `LATIN SMALL LIGATURE` **and** every decomposition target is ASCII | `<compat>`+LIGATURE+ASCII-target = 8 (incl. capital U+0132); `LATIN SMALL LIGATURE` named = 8 by prefix over decomposition-bearing rows (A4: file-wide by prefix 9, incl. decomposition-free U+0153); ASCII-target admits **7** |
| enclosed | name starts `CIRCLED DIGIT` | `<circle>` rows = 240; `CIRCLED DIGIT` admits **10** |
| fractions | name starts `VULGAR FRACTION` | `<fraction>` rows = 20; admits **19** |
| superscripts | tag `<super>` **and** target is a single ASCII digit | `<super>` rows = 249; admits **10** |
| roman | name starts `ROMAN NUMERAL` | `ROMAN NUMERAL` named = 32 by substring over decomposition-bearing rows (A4: 16 uppercase + 16 small; file-wide 24 by prefix, 40 by substring); admits **16** |
| katakana | name starts `HALFWIDTH KATAKANA LETTER` | `HALFWIDTH KATAKANA` named = 59; admits **55** |

Total frozen population: **117 characters**, classes
pairwise disjoint (executed). Narrowing reasons, per
class, stated rather than hidden: circled *digits* (not
the 240 `<circle>` rows) because the marker field (§3)
needs a uniform one-character reading token and circled
letters would collide with the ligature/roman letter
fields; superscript *digits* (not the 249 `<super>`
rows, which are mostly phonetic modifier letters — a
different axis) for the same uniformity; *uppercase*
Roman numerals only — the 16 `SMALL ROMAN NUMERAL` rows
are excluded because lowercase roman is the classic
enumeration marker shape this design must keep out (§3
validator), and because case-duplicate readings would
make the table non-injective against the uppercase set;
katakana *letters* only — the 4 excluded
`HALFWIDTH KATAKANA` rows are U+FF65 MIDDLE DOT, U+FF70
PROLONGED SOUND MARK, and the U+FF9E/FF9F sound marks,
which are punctuation/marks, not letters, and the marks
are combining behaviour, not standalone foldables. Named
exclusions inside otherwise-admitted families: **U+FB05**
(LATIN SMALL LIGATURE LONG S T) fails the ligature
ASCII-target clause — its file target is `U+017F U+0074`
(the long s is non-ASCII; its NFKC `st` is a two-stage
fold, not the file target) — and **U+215F** FRACTION
NUMERATOR ONE fails the fraction name clause (its
putative reading `1/` is not a fraction value). U+0132
(capital IJ ligature) fails the SMALL clause.

**Considered exclusions (A6).** The file carries mirror
families that satisfy most of the superscripts' stated
properties and are excluded by the class list, disclosed
here so the freeze is a decision, not an oversight: the
`<sub>` subscript digits U+2080–2089 (10 members
satisfying every property the superscript clause states
except the tag itself; 64 `<sub>` rows in total) — the
slate's class list is superscripts, and a subscript
field would be a seventh class and a population change,
not an amendment; parenthesized digits (9 rows) —
multi-character readings break the one-character field
uniformity the marker sentence relies on; circled
numbers (41 rows) — two-digit readings, same reason.
The population does not move.

### The reading rule (the semantic core)

For each class, the frozen substitution tables below map
treated character → **reading string**. The control
content (§3) is the treated content with the table
applied; a treated arm is the control with exactly one
field's reading replaced by its character. Per entry,
the table states whether the reading equals raw NFKC
output. The rule per class:

- **ligatures, enclosed, superscripts, roman, katakana:**
  reading == NFKC(ch) == the pinned file's target string,
  for **every** entry.
- **fractions (all 19 entries diverge):** reading ==
  NFKC(ch) with U+2044 FRACTION SLASH replaced by U+002F
  SOLIDUS. Raw NFKC(½) = `1⁄2` (digit, U+2044, digit);
  the reading is `1/2`. Justification: the reading is
  the *plain-text transcription* of the typeset
  character — what a paste into an ASCII prompt means
  and what a human transcriber types; UAX #15's own
  Figure 2 schematises the fraction fold as `¼ → 1/4`
  with a solidus. The deliberate consequence is
  disclosed, not hidden: for the fraction arm,
  NFKC(treated) ≠ control (it is the fraction-slash
  string, a third form), so the pair is
  *reading-equivalent under the frozen table*, not
  NFKC-equal; §9 R7 pre-registers the reading. Every
  diverging entry is listed: **all 19 fraction members**
  (tables below) and **no other member of any class**
  (98 entries, all reading == NFKC, executed).

**Where katakana may live.** The katakana reading is a
*fullwidth katakana character* (non-ASCII) — the one
class whose reading is not ASCII. Following
canonical-form invariant 2's pattern (decomposables live
in carrier/topic/question only, never in expected
values): katakana characters — in **both** forms — live
**only in the marker sentence's kana field** (§3). Every
expected value is pure ASCII (asserted on all seeds);
the topic, the fact sentence, the question, and both
neutrals are pure ASCII; the control content's **only**
non-ASCII code point is the drawn kana reading (asserted
per case). The katakana arm therefore measures the width
fold (halfwidth ↔ fullwidth within katakana), never an
ASCII↔katakana substitution, and it cannot contaminate
the scorer pin.

### Frozen tables (codepoint order — the order §3's draws index)

**Ligatures (7)** — all readings equal raw NFKC:

| code point | char | reading |
|---|---|---|
| U+0133 | ĳ | ij |
| U+FB00 | ﬀ | ff |
| U+FB01 | ﬁ | fi |
| U+FB02 | ﬂ | fl |
| U+FB03 | ﬃ | ffi |
| U+FB04 | ﬄ | ffl |
| U+FB06 | ﬆ | st |

**Enclosed alphanumerics (10)** — all readings equal raw NFKC:

| code point | char | reading |
|---|---|---|
| U+2460 | ① | 1 |
| U+2461 | ② | 2 |
| U+2462 | ③ | 3 |
| U+2463 | ④ | 4 |
| U+2464 | ⑤ | 5 |
| U+2465 | ⑥ | 6 |
| U+2466 | ⑦ | 7 |
| U+2467 | ⑧ | 8 |
| U+2468 | ⑨ | 9 |
| U+24EA | ⓪ | 0 |

**Superscripts (10)** — all readings equal raw NFKC:

| code point | char | reading |
|---|---|---|
| U+00B2 | ² | 2 |
| U+00B3 | ³ | 3 |
| U+00B9 | ¹ | 1 |
| U+2070 | ⁰ | 0 |
| U+2074 | ⁴ | 4 |
| U+2075 | ⁵ | 5 |
| U+2076 | ⁶ | 6 |
| U+2077 | ⁷ | 7 |
| U+2078 | ⁸ | 8 |
| U+2079 | ⁹ | 9 |

**Roman numerals (16)** — all readings equal raw NFKC:

| code point | char | reading |
|---|---|---|
| U+2160 | Ⅰ | I |
| U+2161 | Ⅱ | II |
| U+2162 | Ⅲ | III |
| U+2163 | Ⅳ | IV |
| U+2164 | Ⅴ | V |
| U+2165 | Ⅵ | VI |
| U+2166 | Ⅶ | VII |
| U+2167 | Ⅷ | VIII |
| U+2168 | Ⅸ | IX |
| U+2169 | Ⅹ | X |
| U+216A | Ⅺ | XI |
| U+216B | Ⅻ | XII |
| U+216C | Ⅼ | L |
| U+216D | Ⅽ | C |
| U+216E | Ⅾ | D |
| U+216F | Ⅿ | M |

**Vulgar fractions (19)** — every reading diverges from
raw NFKC by exactly the U+2044 → U+002F rule and nothing
else:

| code point | char | reading |
|---|---|---|
| U+00BC | ¼ | 1/4 |
| U+00BD | ½ | 1/2 |
| U+00BE | ¾ | 3/4 |
| U+2150 | ⅐ | 1/7 |
| U+2151 | ⅑ | 1/9 |
| U+2152 | ⅒ | 1/10 |
| U+2153 | ⅓ | 1/3 |
| U+2154 | ⅔ | 2/3 |
| U+2155 | ⅕ | 1/5 |
| U+2156 | ⅖ | 2/5 |
| U+2157 | ⅗ | 3/5 |
| U+2158 | ⅘ | 4/5 |
| U+2159 | ⅙ | 1/6 |
| U+215A | ⅚ | 5/6 |
| U+215B | ⅛ | 1/8 |
| U+215C | ⅜ | 3/8 |
| U+215D | ⅝ | 5/8 |
| U+215E | ⅞ | 7/8 |
| U+2189 | ↉ | 0/3 |

**Halfwidth katakana (55)** — all readings equal raw
NFKC (each reading is the fullwidth katakana character):

| code point | char | reading |
|---|---|---|
| U+FF66 | ｦ | ヲ |
| U+FF67 | ｧ | ァ |
| U+FF68 | ｨ | ィ |
| U+FF69 | ｩ | ゥ |
| U+FF6A | ｪ | ェ |
| U+FF6B | ｫ | ォ |
| U+FF6C | ｬ | ャ |
| U+FF6D | ｭ | ュ |
| U+FF6E | ｮ | ョ |
| U+FF6F | ｯ | ッ |
| U+FF71 | ｱ | ア |
| U+FF72 | ｲ | イ |
| U+FF73 | ｳ | ウ |
| U+FF74 | ｴ | エ |
| U+FF75 | ｵ | オ |
| U+FF76 | ｶ | カ |
| U+FF77 | ｷ | キ |
| U+FF78 | ｸ | ク |
| U+FF79 | ｹ | ケ |
| U+FF7A | ｺ | コ |
| U+FF7B | ｻ | サ |
| U+FF7C | ｼ | シ |
| U+FF7D | ｽ | ス |
| U+FF7E | ｾ | セ |
| U+FF7F | ｿ | ソ |
| U+FF80 | ﾀ | タ |
| U+FF81 | ﾁ | チ |
| U+FF82 | ﾂ | ツ |
| U+FF83 | ﾃ | テ |
| U+FF84 | ﾄ | ト |
| U+FF85 | ﾅ | ナ |
| U+FF86 | ﾆ | ニ |
| U+FF87 | ﾇ | ヌ |
| U+FF88 | ﾈ | ネ |
| U+FF89 | ﾉ | ノ |
| U+FF8A | ﾊ | ハ |
| U+FF8B | ﾋ | ヒ |
| U+FF8C | ﾌ | フ |
| U+FF8D | ﾍ | ヘ |
| U+FF8E | ﾎ | ホ |
| U+FF8F | ﾏ | マ |
| U+FF90 | ﾐ | ミ |
| U+FF91 | ﾑ | ム |
| U+FF92 | ﾒ | メ |
| U+FF93 | ﾓ | モ |
| U+FF94 | ﾔ | ヤ |
| U+FF95 | ﾕ | ユ |
| U+FF96 | ﾖ | ヨ |
| U+FF97 | ﾗ | ラ |
| U+FF98 | ﾘ | リ |
| U+FF99 | ﾙ | ル |
| U+FF9A | ﾚ | レ |
| U+FF9B | ﾛ | ロ |
| U+FF9C | ﾜ | ワ |
| U+FF9D | ﾝ | ン |

## 3. Frozen generated material

**The hard problem, stated.** The sibling panel hosts
*one* treatment axis per case (an insertion; a
substitution in the topic and value words). No natural
fact sentence hosts all six classes at once: no English
carrier word contains a ligature, a circled digit, a
vulgar fraction, a superscript, a Roman numeral, and a
halfwidth katakana character, and the expected value
cannot host any of them (scorer pin). Splitting the
panel by class (4 cases per class) destroys the
classification: a per-arm population of 4 can never
reach any honest MIN_D. The solution is this design's
central structural decision, made explicitly so it can
be attacked as such: **every case carries a frozen
marker sentence hosting exactly one field per class**,
so every treated arm treats every case, at a uniform
dose, and every per-arm paired population is the full
24.

- **Cases:** 24, stage-local, generated only, never the
  user's Split. Case *i*: topic = `TOPICS[i]` from the
  siblings' frozen 24-word ASCII list (harbor, summit,
  quarry, tundra, granite, willow, copper, saddle,
  walnut, velvet, cobalt, marble, cinder, falcon,
  garnet, saffron, indigo, juniper, lagoon, magnet,
  nickel, pepper, quartz, birch). Value = `WORDS[draw] +
  "-" + number`, WORDS the canonical-form 24-word ASCII
  pool, number ∈ [1000, 9999], pairwise distinct across
  cases by redraw. Two distinct neutrals from density's
  frozen `NEUTRAL_STATEMENTS` (24 ASCII statements).
  Fact = `The {topic} inspection code is {value}.`
  Question = `What is the {topic} inspection code?`
  Content = `{neutral1} {fact} {marker} {neutral2}
  {question}`. Expected = value.
- **Marker sentence (frozen template, verbatim):**
  `The marker line reads script {f0}, circle {f1}, share {f2}, power {f3}, regnal Henry {f4}, kana {f5}.`
  Reading fields: f0 = the drawn ligature's reading;
  f1 = the drawn enclosed member's reading; f2 = the
  drawn fraction's reading; f3 = `x` + the drawn
  superscript's reading; f4 = the drawn Roman member's
  reading; f5 = the drawn katakana member's reading
  (fullwidth). In a treated arm, exactly that class's
  field is its frozen character instead (superscript:
  `x` + its character); the other five fields stay in
  reading form. The marker sentence sits immediately
  after the fact sentence and before the second neutral
  — inside the payload core, not in the neutral fluff —
  and it is *carrier* text: the question never references
  it (validator below), so its fields are content the
  model reads past on the way to the answer, exactly the
  position typeset paste occupies in a real prompt.
- **Tasks: recall/extraction only.** The question asks
  only for the inspection code. **No arithmetic is ever
  performed or requested on fraction (or any) values**
  (A7): the screen's operator set is `+`, `*`, and the
  six arithmetic words `plus`, `times`, `sum`, `total`,
  `add`, `subtract`, `multiply`, `divide` (word-boundary
  regex `\b(plus|times|sum|total|add|subtract|multiply|divide)\b|\+|\*`
  must not match fact+marker+question), and the
  guarantee is stated in its true form: no arithmetic is
  *requested* — the question is byte-identical to the
  frozen template and references no marker field; the
  fraction reading's solidus is a transcription
  character inside an unreferenced field, disclosed, not
  an operator in any task. The fraction field's label is
  `share` — a label token, not a quantity the task uses.
- **Roman numerals are content, never enumeration — by
  validator, not authorial care.** The Roman field is
  the regnal suffix in `regnal Henry {f4}` — a name, in
  the middle of a sentence. Executed screens on every
  generated content (all arms, all seeds): (i) the
  content contains no newline at all, so no line-initial
  marker position exists; (ii) no sentence's first token
  is a Roman reading or a Roman character; (iii) the
  marker contains the literal `regnal Henry `
  immediately before the Roman field; (iv) by
  construction of the field list, Roman
  characters/readings occur at no other position — the
  fact and question templates contain only topic and
  value (topic ∈ the lowercase ASCII list, value ∈ the
  ASCII word-number grammar). The excluded SMALL ROMAN
  set (§2) removes the lowercase marker shape from the
  population itself.
- **Scorer pin:** every expected value is pure ASCII
  (asserted on all seeds); verdicts are `core.evaluate`'s
  byte-exact `answer == case.expected` (the `exact_match`
  conjunction in `core.py`), conjoined with boundary
  presence as in every sibling stage. Compat characters
  live only in the marker sentence (and, for katakana
  readings, only in its kana field — §2); a model echoing
  a treated field into its answer is separated by §5.
- **Draws — conversion pinned in text:** material
  `f"{seed}|compatfold|{index}|{slot}|{counter}"`;
  integer `int.from_bytes(sha256(material).digest()[:8], "big")`;
  slot 0: word = `WORDS[draw % len(WORDS)]`; slot 1:
  number = `1000 + draw % 9000`; slots 2/3: neutrals =
  `NEUTRAL_STATEMENTS[draw % len(NEUTRAL_STATEMENTS)]`
  (slot 3 advances +1 mod len on collision); slot 4:
  ligature = `LIGATURES[draw % len(LIGATURES)]`; slot 5:
  enclosed; slot 6: fraction; slot 7: superscript;
  slot 8: roman; slot 9: katakana — each
  `draw % len(<class list>)` over the §2 tables **sorted
  by code point**. A value collision redraws **slots 0+1
  only** at counter+1 (slots 2–9 stay at counter 0).
  **Give-up counting (A2), pinned exactly as the module
  implements it:** the redraw counter starts at 0 and
  increments once per colliding redraw; exhaustion is
  raised when the counter exceeds 1000 (i.e. after 1001
  increments the stage fails closed), with stage detail
  `compatibility_form_screen_failed`. Namespace is
  **new** (`compatfold`): each shipped probe's material
  names its own probe, so panels are independent draws.
  *Latent-path disclosure:* value collisions on the five
  shipped evidence seeds = **0** (executed in the design
  round), so the redraw semantics are text-pinned but
  evidence-latent; the build's freeze tests force a
  collision (§13 debt 3).
- **Panel-size derivation (executed in the design
  round):** under this probe's own null and floors (§6),
  exact marginal power at the 25-point target is N=16:
  **0.3904**, N=20: **0.6395**, N=24: **0.8159** — 24 is
  the smallest of the three house sizes reaching the
  design's 0.80 criterion, and the shared sibling size.
  The target effect (25 points) is the decision
  threshold of §1's utility: an ingest-normalization
  decision does not turn on a 3-point wobble; smaller
  effects read `below_threshold` (§9), never "no
  effect".

**Generator determinism evidence (design round).** The
designer's generator derives the class tables from the
pinned file (no hand-copied table) and implements the
frozen spec twice — implementation A by whole-field
substitution through the reading→character table on the
control fields, implementation B by direct row selection
— and asserts byte-identical populations for seeds
{0, 1, 7, 42, 20261030}: **A == B on all five seeds**. A
second implementation written from this text alone by
the review's red team B is byte-identical on all 7 arms
× 24 cases × the same five seeds (840 cells, 0
mismatches). Byte-divergence: every treated arm differs
byte-wise from control on **24/24 cases, every seed**.
Dose: exactly **1 frozen code point per case per treated
arm** (min = max = 1, every arm, every seed) — and, by
the review's stronger whole-content execution over 60
seeds, the foldable-character difference between control
and any treated content is exactly 1 over the *whole
prompt*: every other field's reading is NFKC-stable, so
no background foldable leaks into the dose. Per-arm
length deltas vs control, over the five seeds (executed):
ligature codepoint Δ ∈ [−2, −1], UTF-8 byte Δ ∈ [0, 1];
enclosed Δcp 0, Δbyte +2; fraction Δcp ∈ [−3, −2],
Δbyte ∈ [−1, 0]; superscript Δcp 0, Δbyte ∈ [1, 2];
roman Δcp ∈ [−3, 0], Δbyte ∈ [−1, 2]; katakana Δcp 0,
Δbyte 0 (both kana forms are 3 UTF-8 bytes). Distinct
members drawn per class over 5 seeds × 24 cases:
ligature 7/7, enclosed 10/10, fraction 19/19,
superscript 10/10, roman 16/16, katakana 52/55 — the
draw reaches (nearly) the whole frozen table, so no
class verdict can rest on one cherry-picked character.
The question is byte-identical across arms for every
case (asserted), and all §3 validators pass on every
case/arm/seed. Example (seed 0, case 0): control marker
`... script fi, circle 7, share 1/7, power x9, regnal Henry V, kana カ.`;
the fraction arm replaces `1/7` by `⅐`, the katakana arm
replaces `カ` by `ｶ`, everything else byte-identical.

## 4. Arms, sites, dose

**Arms (7, fully crossed):** `control` (all six marker
fields in reading form) plus one treated arm per class —
`ligature`, `enclosed`, `fraction`, `superscript`,
`roman`, `katakana` — each replacing exactly its own
field in every case. There is no arm in which two
classes are treated at once: the per-class verdict is
the §1 decision unit, and a combined arm's fire could
not be attributed to a class.

**Sites.** Substitution is inside the **marker sentence
only**, at the class's frozen field position (field
index 0–5 in the field list — a positional rule, not a
search-replace: several readings, e.g. a digit, also
occur inside the value, and the site is the field, never
the first textual occurrence). The fact sentence,
question, and neutrals are byte-identical across arms by
construction (asserted for the question; the
fact/neutrals are arm-invariant inputs to the content
assembly).

**Dose, stated honestly.** Exactly 1 frozen code point
per case per treated arm — the siblings' single-site
dose, uniform across all six classes, which is what
makes cross-arm comparisons dose-matched in count (their
codepoint/byte deltas differ by class and are printed in
§3). The treated field is one field of a carrier
sentence, not the answer-bearing word: the reading
frozen in advance of a null is therefore "one compatibility-form
field in the carrier sentence adjacent to the fact did
not move code recall at the frozen floors" — not
"compatibility forms in answer-bearing text are safe"
(§9 R3). Dose escalation in response to a null is
forbidden (§9 R1).

**Budget (house pattern):** 24 cases × 7 arms × up to
24 candidates = **4,032 generations** worst case; 576
ground records; house 12 MiB records cap.

## 5. Decomposition (ordered decision list; total)

The byte-exact scorer is the instrument Hua et al.
(arXiv:2509.01790) warn about; no discordance is
reported undecomposed. Classes are an **ordered decision
list** — the literal predicates overlap, so order is
part of the definition: each class is reached only when
every earlier predicate fails.

**Cell decomposition** — exactly one class per completed
cell:

1. `pass` — verdict passed.
2. `boundary_only` — answer digest == expected digest;
   the boundary conjunct alone failed.
3. `wrong_value` — the answer parses under the frozen
   value grammar `^[a-z]+-[0-9]{4}$` with word ∈ WORDS
   and number ∈ [1000, 9999], and is not the expected
   value. The grammar is applied with fullmatch
   semantics: a trailing newline (or any other trailing
   character) makes the answer out-of-grammar — under a
   plain `re.match`, the `$` anchor also matches before
   a final newline, which would admit the expected
   value plus newline into this class; the battery
   carries that witness (cycle-231 audit F1). **The
   only class counting toward the headline
   J (§6).** Scope disclosed: J counts wrong values
   *inside* the frozen grammar; a value change outside
   it (out-of-pool word, leading-zero number) decomposes
   as `non_answer`, is printed per arm, and never counts
   toward J; the OC's q is the **in-grammar** wrong-value
   rate wherever it appears.
4. `echo_compat` — answer ≠ expected **and**
   `NFKC(answer) == expected` **and** the answer contains
   at least one character from **the arm's own frozen
   class table**. The model rendered the expected value
   using the treated class's forms (e.g. under the
   enclosed arm, `alder-①②③④` for `alder-1234`).
   **(A3)** The control arm's class-4 table is the
   **empty set**: class 4 never fires for control cells,
   and a folded control answer classifies `form_only`.
5. `form_only` — answer ≠ expected **and**
   `NFKC(answer) == expected` (and no arm-table
   character, or class 4 would have taken it).
   **(A5(i))** Classes 4+5 together are exactly the
   answers whose NFKC fold *is the expected value*,
   split by whether the treated class supplied the
   forms. Case-changing renderings (circled capitals
   folding to uppercase, squared letters) fold to a
   *different* string and decompose `non_answer`, never
   toward J. The breadth is intended: no
   compatibility-rendered correct value is ever counted
   as a value change, whichever class produced it. The
   cross-arm consequence is stated plainly: `form_only`
   rates are comparable across arms (same predicate),
   while `echo_compat` is arm-relative by definition.
6. `non_answer` — everything else, including empty
   answers, the marker token itself as an answer (`½`),
   a halfwidth-kana answer (it folds to fullwidth kana,
   never to an ASCII value), and out-of-grammar value
   changes (class 3's scope note).

**Echo reachability, executed (A5(iii)).** The
design-phase probe searched two numbers (1234/9876); the
exhaustive confirmation — a single-character substitution
search over the whole value space, every WORDS word ×
the full number range = 216,000 expected values — was
executed in the review round by both red teams, and the
build's decomposition battery re-executes it as a test
(§13 debt 6): `echo_compat` is reachable **only for the
enclosed and superscript arms** (their readings are
digits, and digits occur in every expected value). For
ligature, fraction, roman, and katakana it is
**unreachable by construction**: no WORDS word contains
a ligature reading as a substring; a fraction reading
contains `/`, which no expected value contains; Roman
readings are uppercase and fold to uppercase, never to
the lowercase expected words; kana readings are
non-ASCII and fold to fullwidth kana. This is printed
here, not discovered later: for those four arms the
compatibility-rendering split is carried entirely by
`form_only`, and any cross-arm reading of echo rates
must use this paragraph (§8).

**Grammar battery (frozen wrong shapes the tests pin)**
— 15 executed cases (A5(ii), enumeration complete): the
correct shape (`pass` — the precedence witness: the
exact expected answer satisfies the literal `form_only`
predicate and must land in class 1, not 5); a
digest-equal boundary failure (`boundary_only` before
all form classes); a *wrong value in the right form* and
*another case's value* (`wrong_value`); the enclosed-arm
digit echo and the superscript-arm digit echo
(`echo_compat`); the superscript echo **under the
ligature arm** (`form_only` — the cross-arm witness); a
fullwidth value (`form_only`); the marker token as
answer, an out-of-pool word, a leading-zero number, an
empty answer, a short number, a trailing-space answer
(all `non_answer`); the expected value plus a trailing
newline (`non_answer`, cycle-231 audit F1); and the
halfwidth-kana answer
(`non_answer`). The battery additionally carries A3's
control witness: a fullwidth rendering of the expected
value under the control arm → `form_only`. A
20,000-answer fuzz in the design round and a 50,000-answer
adversarial fuzz in the review round confirm totality
(every answer receives exactly one class, no exceptions).

**Pair decomposition** — for discordant pairs, the
sibling total partition: answer digests equal →
`boundary_only`; digests differ, both boundaries found
→ `exact_only`; otherwise → `mixed`. Cross-tabulated
with the failing cell's class.

## 6. Headline statistic, decision vocabulary, floors, guard

**Regime gate first.** Id divergence is measured at
encode time on the FULL render id sequence per (case,
candidate), per arm against control (canonical-form
invariant 9 discipline). Per-arm reporting:
`n_ids_identical`, `n_ids_divergent`, the per-case
`id_length_delta` list, `max_abs_id_delta`, and
`n_same_ids_verdict_flips`, per (candidate, arm).
**Pre-registered regime expectation** (stated as
expectation, discovered by measuring ids, never asserted
per checkpoint): SentencePiece's default rule is
`nmt_nfkc`, "a custom variant of Unicode NFKC", applied
*before* tokenization, with the compiled rule embedded
in the model file and an `identity` alternative
(SentencePiece normalization doc — §14). The same doc
warns the implementation "does not implement the
entirety of Unicode NFKC". So: families whose pipeline
applies an NFKC-class rule are expected to land
ids-identical for the classes their compiled rule folds
— plausibly most classes, but the partial-implementation
caveat makes per-class divergence an open empirical
question (a rule that folds enclosed digits but not
halfwidth katakana is exactly the kind of split the
per-arm gate exists to discover); families on `identity`
rules or byte-level tokenizers without a normalizer are
expected to encode the treated fields **divergently**.
Note the fraction subtlety (§2): a true NFKC normalizer
folds `½` to the fraction-slash string `1⁄2`, which is
*neither* the treated form *nor* the control reading —
the gate still decides it correctly, because it compares
the two arms' renders to each other, and a normalizer
that maps the treated field to a third form while
leaving the control's `1/2` untouched produces divergent
ids (a real, reportable finding: the pipeline rewrites
the foldable to something the ASCII reading is not).

**Headline population** per (candidate, arm): cases
completed in both control and arm **and** id-divergent
for that arm. Ids-identical arms are reported
instrument-inapplicable per family, never as passes.
D = control passes in the population; J = of those, arm
cells decomposed `wrong_value`.

**Decision vocabulary (total function of D, J, and the
guard)** — the tree's existing sibling set only, no
invented words:

- `insufficient` — D < 10, **or** the guard withholds.
- `no_discordance_observed` — D ≥ 10, J = 0, guard
  passes.
- `discordant_asymmetric` — D ≥ 10, J ≥ 6, guard
  passes. (**Fires**: the §7 invariance claim is
  falsified for this arm.)
- `below_threshold` — D ≥ 10, 0 < J < 6, guard passes.

The population is conditioned on control passes (the
negation / instruction-distance / history-frame /
invisible-chars D-conditioning precedent), so
discordance has one direction by construction and
`discordant_balanced` is structurally unreachable.
Reverse discordants are counted and printed beside every
label as `n_reverse`, never pooled into J. The
two-directional canonical rule was rejected by execution
in the sibling rounds (transplanted floors fire at
0.8897 on the competent independent null at N = 24);
this design does not re-litigate it. **No sign test is
used anywhere** (the tree's `density.sign_test_p` is
two-sided and not completion-monotone — round-1
settlement); a descriptive McNemar figure, if printed,
uses `confidentiality.mcnemar_exact_p` over completed
pairs only and is never a decision input.

**Floors — derived under this probe's own null, frozen
here.** `MIN_D = 10`, `MIN_J = 6`, from this probe's
floor search (design round: exact enumeration over
D ~ Bin(24, pc), J|D ~ Bin(D, q); grid MIN_D ∈ 1..16 ×
MIN_J ∈ 2..8, requiring competent-null fire ≤ 0.02 and
maximal target power). MIN_J = 6 is set by the search:
every MIN_J ≤ 5 pair breaches the null bound at every
MIN_D (minimum null at MIN_J = 5 over all MIN_D =
0.059095). MIN_D = 10 is **selected from a plateau**: at
printed (4 dp) precision the competent-null and target
figures are identical for every MIN_D ∈ [1, 14]; in
exact arithmetic both figures are exactly constant over
MIN_D ∈ [1, 6] and (10, 6) trails the exact maximum
target power by 2.796189e-11 — stated as a plateau
selection, not a point derivation. No floor is borrowed:
the search was run against this probe's rule and null,
and its output happens to land on the siblings' pair.

**Operating characteristics** (fire =
`discordant_asymmetric`). **Convention:** the independent
column is **exact enumeration under the marginal
convention** — per-case independence; **q = the arm's
marginal in-grammar wrong-value rate among
control-passing cases**, with each row's pc, pa (the
arm's marginal pass rate), and q printed so the row is
checkable. The paired column is **exact** (bivariate
normal latents, ρ = 0.5, Simpson quadrature + trinomial
sum) — not Monte Carlo.

| world | independent (exact, marginal) | paired ρ=0.5 (exact) |
|---|---|---|
| competent null (pc .90, q .10) | 0.0173 (0.017274) | 0.0044 (0.004442) |
| weak null (.80 / q .20) | 0.1743 (0.174277) | — |
| random labeller (.50 / q .50) | 0.5503 (0.550306) | — |
| drop 10pt (.90 / pa .80, q .20) | 0.2542 (0.254234) | — |
| **drop 25pt (.90 / pa .65, q .35) — target** | **0.8159 (0.815868)** | **0.6981 (0.698078)** |
| drop 25pt at base .85 (.85 / pa .60, q .40) | 0.8761 (0.876145) | — |
| drop 50pt (.90 / pa .40, q .60) | 0.9990 (0.999044) | — |
| sensitivity: 18/24 paired, drop 25pt | 0.5214 (0.521448) | — |
| sensitivity: 18/24 paired, competent null | 0.0038 (0.003803) | — |

The weak-null (0.1743) and random-labeller (0.5503) rows
are printed, not hidden: the label is comparative
evidence at a stated base rate, and each candidate's
control rate is printed beside its label so a reader can
see which null row applies.

**M-conditioned operating characteristics (A1).** The
rows above condition on the full panel (M = 24). When
the ids gate shrinks an arm's divergent domain to M,
the same exact enumeration at panel size M gives
(drop-25 / drop-10 / competent null):

| M | drop-25 | drop-10 | competent null |
|---|---|---|---|
| 24 | 0.8159 | 0.2542 | 0.0173 |
| 20 | 0.6395 | 0.1356 | 0.0068 |
| 18 | 0.5214 | 0.0889 | 0.0038 |
| 16 | 0.3904 | 0.0527 | 0.0019 |
| 12 | 0.1379 | 0.0113 | 0.0003 |
| 10 | 0.0331 | 0.0022 | 0.0001 |

The direction is safe for fires (the competent-null fire
rate only *falls* as M shrinks — the licence is not
overstated), but an acquittal at small M is materially
weaker evidence than the M = 24 rows price; that is why
every report prints the arm's divergent-domain size M
(the headline record's `n_ids_divergent`) beside its
label (§10, reporting), and §7's caveat is conditioned
on M.

**Family-wise price** — counting **every arm that
receives a label: all 6 treated arms** × 24 candidates =
**144 comparisons**, from unrounded rates:

| null | per-comparison (unrounded) | E[false labels] | P(≥ 1) |
|---|---|---|---|
| competent independent | 0.017274 | 2.4875 | 0.9187 |
| paired ρ=0.5 (exact) | 0.004442 | 0.6397 | 0.4733 |
| random labeller | 0.550306 | 79.2441 | 1.0000 |

Printed beside the licence in every report.

**Completion guard + incompletion perimeter.** Two
channels, as settled in rounds 1–2 and transferred with
their amendments folded in — including the ids dimension
in the design-phase evidence itself:

- *(a) Hidden slots* — completed generations whose span
  does not parse are completed cells, decomposed
  `non_answer` (§5), **inside** the perimeter;
  `n_non_answer` per arm is printed, and §9's
  reading frozen in advance governs high-non-answer
  discordance. They never count toward J.
- *(b) Incompletion* — a generation producing no
  boundary-delimited span at all (or a missing cell) has
  no cell; the case leaves the paired population. The
  guard brings it inside the decision perimeter by
  construction:
  - **Domain pinned:** the worst-case completion and the
    cap range over exactly the incomplete cases **in the
    pair's population domain** — cases whose control and
    arm renders are id-divergent for that arm (an
    encode-time property, decidable whether or not the
    generation completed) and which have at least one
    missing cell. Ids-identical cases never enter the
    completion, in either direction.
  - *Fire cannot be manufactured:* J and D are absolute
    counts over completed cases; no completion raises
    observed J, and the guarded fire uses observed
    counts only.
  - *Acquittal cannot be manufactured:* an acquittal
    label (`no_discordance_observed` / `below_threshold`)
    issues **only if stable under the worst-case
    completion** — every incomplete case in the domain
    completed as control-pass + arm `wrong_value` (D+1,
    J+1 each). The (D+1, J+1) counting is a blanket
    over-count: it also ranges over
    observation-inconsistent completions for
    partially-observed cases (a case with an observed
    control-fail cell is counted as completing
    control-pass) — deliberately conservative,
    disclosed. Consequences disclosed: fires issue on
    observed counts (incompletion present does not by
    itself withhold a fire); `no_discordance_observed`
    issues only at n_incomplete = 0; `below_threshold`
    survives only while J + n_incomplete < 6.
  - **Cap (A1 restated):** if incomplete cases in the
    domain exceed **6** for a pair, the label is
    `insufficient` regardless. The cap bounds incomplete
    cases *within the domain* at 6; it guarantees
    paired ≥ 18 only when the domain is the full 24
    cases (paired = M − k). When the ids gate shrinks
    the domain to M, labels issue at paired counts down
    to M − 6 (and to M at k = 0), with the operating
    characteristics conditioned on M as tabled above.
    The cap is a paired-population quality floor; its
    only marginal effect in the exhaustive audit below
    is suppressing completion-stable fires, disclosed as
    such.
  - *Executed attack on this guard* (design round):
    **exhaustive (D, J, k) audit** over the **reachable**
    state space (D + k ≤ 24 — a pair's domain has 24
    cases) — **0 manufactured-fire violations, 0
    unstable-acquittal violations**; fires issue at every
    k ∈ [0, 6]; the cap suppresses **264**
    completion-stable fire states at k > 6 over
    reachable states (its only marginal effect). **MC
    worlds with the ids dimension executed** (each case
    id-divergent w.p. 0.8 — a simulation parameter,
    stated, not a claim about any checkpoint;
    non-divergent cases are excluded from the population
    *and* from k; 10,000 trials per world, seed
    20261030): **World A** (zero-effect null; incompletion
    concentrated on concordant pass/pass cells, removed
    w.p. 0.5 — the fire-manufacture attack): guarded
    fire **0.0037** vs unguarded 0.0069 (manufacture
    fails); guarded acquittals collapse into withholding
    (`insufficient` 0.9765). **World B** (true drop,
    q = 0.50; 60% of would-be J cells removed — the
    acquittal-manufacture attack): unguarded fire 0.1170
    with `below_threshold` at **0.7231 — a manufactured
    acquittal**; guarded fire 0.1046 with
    `below_threshold` at **0.0763** and 0.8191 withheld.
    The residual 0.0763 is the pinned domain working as
    specified: an ids-identical incomplete case *cannot*
    complete into the population, so it correctly does
    not count against stability; the displaced mass is
    still overwhelmingly withholding, never acquittal at
    anywhere near the unguarded rate. **World C** (same
    drop, flat incompletion 0.15): guarded fire
    **0.7719** vs unguarded 0.7824 — the guard costs
    little when incompletion is not sign-concentrated.
    (World fire levels sit below the §6 OC rows because
    the simulated divergent share (~19.2 of 24 cases)
    shrinks the effective panel — a property of the
    stated simulation parameter, printed so the worlds
    are not misread as OC.)
  - *Cost, printed:* in high-incompletion worlds the
    guard withholds aggressively (World A guarded
    `insufficient` 0.9765). That is the chosen direction:
    incompletion buys silence, never a *manufactured*
    verdict in either direction — where the precise
    label behaviour is the bullets above, not the
    slogan.

## 7. Pre-registered falsifiable claim

> Over generated fact-recall cases (§3) whose control
> and treated renders are **id-divergent** (§6 gate), for
> a given candidate and compatibility class arm, the
> arm's verdicts on control-passing cases show no
> in-grammar value-changing discordance at or above the
> frozen floor: conditioning on the id-divergent paired
> population, **under the §6 guard**, J < 6 whenever
> D ≥ 10.

- **Falsified** for a (candidate, arm) exactly when its
  guarded label is `discordant_asymmetric`. That is the
  whole falsification condition; a fire proves only that
  *this* class's compatibility forms, at one field of
  the carrier marker sentence, moved *this* checkpoint's
  answers to *wrong in-grammar values* on *this*
  generated population — nothing about weights, attacks,
  other characters in the class's broader Unicode
  family, or compatibility forms in answer-bearing
  positions. **(A8)** For the fraction arm under an
  NFKC-class normalizer, the form that reached the model
  is the pipeline's rewritten `1⁄2` form (§6), and a
  fire is attributable to that rewritten form against
  the ASCII reading.
- **Not falsified ≠ robust.** `below_threshold` and
  `no_discordance_observed` acquit only at the printed
  OC: **(A1)** at M = 24 a 10-point marginal drop fires
  only 0.2542 of the time; at M = 12, 0.0113.
  Ids-identical arms are instrument-inapplicable for
  that family, reported as such, never as passes. A run
  landing 100% ids-identical has measured the
  tokenizer's normalizer, full stop.
- **No direction is frozen in advance.** §1's prior is
  mechanics, not an effect: whether folding a field
  helps, hurts, or neither is the measurement, and the
  claim is two-sided in substance (the rule is
  one-directional by D-conditioning, with `n_reverse`
  printed — §6). A null result is a legitimate,
  outcome frozen in advance of this design, stated here in
  advance.
- The claim is per-arm and per-candidate. **No pooled
  headline and no winner readout in v1** (sibling
  precedent: the ingest-normalization decision is per
  checkpoint and per class).

## 8. Isolation rows

- **The sibling is the isolating control by
  construction:** canonical-form (§8.54) runs the
  identical instrument shape over canonical equivalence;
  this probe runs it over compatibility equivalence. A
  family that fires here and not there (or the reverse)
  localises the sensitivity to one of the standard's two
  equivalence halves; the §2 boundary (every member
  satisfies NFC == NFD == ch) guarantees the populations
  are disjoint, so neither probe's fire can be the
  other's artefact.
- **Cross-class isolation within the panel:** every
  treated arm leaves the other five marker fields in
  reading form, so each arm's render differs from
  control at exactly one field. A fire is attributable
  to that field's class, not to "non-ASCII text
  somewhere in the prompt" — the six arms are each
  other's background. Joint fires across all six arms
  for one candidate read as "any compatibility-form
  field disrupts this candidate" (a general
  form-fragility finding), printed as such.
- **Dose isolation:** all six arms are dose-matched at
  exactly 1 frozen code point per case (§4), so a fire
  at one class and not another is not explainable by
  dose count; the residual length confound
  (codepoint/byte deltas differ by class) is printed per
  arm in §3 and in the report, and a class whose delta
  is extreme for a case (roman Δcp −3) is visible, not
  pooled away.
- **Echo/form readings are per-arm descriptive:**
  cross-arm comparisons use only the uniformly-defined
  classes (`wrong_value`, `non_answer`, `boundary_only`);
  `echo_compat` is arm-relative and, per §5's executed
  reachability audit, structurally empty for four arms —
  it is never compared across arms.
- Non-target arms/families under the null process: every
  arm notionally treated with a null character behaves
  as the competent/paired rows; the report's per-arm
  table (6 treated arms × candidates, each with label,
  D, J, `n_reverse`, `n_non_answer`) is itself the
  isolation grid.

## 9. Risks and readings frozen in advance

- **R1 — No direct prior; benign effect genuinely
  unknown.** §1 states it: no published study measures
  LLM behaviour on NFKC-foldable characters. Boucher et
  al. measure *attacks* on vulnerable NLP systems;
  nothing here inherits an effect size from them. A null
  is a legitimate outcome frozen in advance — a
  replication-style null in the tree's history-frame
  sense — and **dose escalation in response to a null is
  forbidden**: no "treat all six fields at once and
  retry", no "move the fields into the value word"
  inside this probe. Either is a new design.
- **R2 — Regime shrinkage.** If most families land
  ids-identical for most arms, the classifiable
  population shrinks; the 18/24-paired sensitivity rows
  (§6) and the A1 M-conditioned rows price this in
  advance. The katakana arm has a specific version of
  this risk: a pipeline whose normalizer handles
  Latin/digit foldables but not width variants (or the
  reverse) makes exactly one arm inapplicable — a
  per-arm finding, reported per arm, never pooled.
- **R3 — Carrier-position attenuation.** The treated
  field is in the carrier marker sentence, not in the
  answer-bearing word (§4's disclosed trade for hosting
  six classes in 24 cases). A null therefore does not
  license "compatibility forms are safe in
  answer-bearing text"; it licenses the §7 claim's exact
  scope. Conversely, a fire from a carrier field is
  *stronger* evidence of form sensitivity than the same
  fire in the fact word would be — the treatment never
  touched the sentence the answer comes from.
- **R4 — Scorer artefact.** Without §5 this probe
  counts a model answering in compatibility forms as a
  capability change; with it, the headline counts only
  `wrong_value`. The residual risk is masking via
  silence (`non_answer`): printed per arm, with the
  reading frozen in advance that an arm whose discordance
  is dominated by `non_answer` is reported as "the
  class's forms disrupt output form", a distinct finding
  — never silently acquitted, never pooled into J.
- **R5 — Expected-value semantics.** The panel never
  lets "is ½ the same answer as 0.5?" arise: no question
  asks for a fraction, a count, a power, or a numeral;
  every expected value is a word-number code determined
  by the fact sentence, and the marker fields are
  unreferenced carrier content (§3 validators).
  Fraction/superscript/Roman classes are restricted to
  recall of an unrelated code precisely so the probe
  measures form, not scorer philosophy about values.
- **R6 — Fully specified panel.** Per Pecher et al.
  (arXiv:2602.04297), underspecification inflates
  apparent sensitivity; the panel's
  instruction/question/contract strings are frozen
  verbatim (§3) so the instrument measures the field's
  form, not ambiguity.
- **R7 — The fraction reading is a transcription
  choice.** §2 discloses it fully: the reading uses the
  solidus; raw NFKC uses U+2044. The third-form
  consequence (a true-NFKC pipeline produces a string
  that is neither arm) is carried in §6 and, per A8, in
  §7's gloss. The alternative (control reading `1⁄2`
  with U+2044) was rejected because the control is
  defined as the plain-text paste a user actually has,
  and an ASCII prompt never contains U+2044 unless a
  normalizer put it there. The other five classes are
  untouched by this question (their readings are raw
  NFKC exactly).
- **R8 — Subset-extension attack.** "This is
  perturbation with extra classes" is the obvious review
  line. The canonical-form design's differentiation
  answers it, and this design keeps the instrument shape
  identical in kind to §8.54, not merely similar:
  generated panel, per-class paired classification,
  encode-time ids gating, digest-committed treatments,
  own-null floors, completion guard. Perturbation's
  operator is a blended noise rate over the `<wide>`
  class this population excludes by rule (§2, §11).

## 10. Fail-closed vocabulary, record shape, naming

- Whole-stage failure: `{"available": False, "detail":
  "compatibility_form_unavailable"}`. Stage details:
  `cancelled`,
  `compatibility_form_record_budget_exceeded`,
  `compatibility_form_screen_failed` (A2). Per-cell
  details: `""`, `arm_context_exceeded`,
  `arm_evaluation_failed`. Rates are None only when the
  denominator is 0. Reports render counts and rates
  only, every payload-derived string through
  `_escape_text`; every headline row prints the arm's
  divergent-domain size M beside its label (A1).
- Record per (candidate, case): fixture id, topic,
  expected digest, per-arm content digests (7), per-arm
  token counts and render-id digests, `ids_identical`
  per arm vs control, `id_length_delta` per arm, the
  frozen-table version pin (UnicodeData 15.0.0 + the
  pinned file's SHA-256 + the derived table's own
  digest), **the six drawn members per case (as code
  points)**, cells and cell details per arm. Cells carry
  `answer_text` bound by `answer_digest` (the cycle-190
  sanctioned pattern, as in the siblings). Content
  digests per arm are mandatory (canonical-form
  invariant 10): the diff validator's mirror regenerates
  all seven arm contents from the seed and checks each
  digest, and re-derives the drawn members from the seed
  rather than trusting the recorded ones. The mirror
  decomposes pairs in the sibling pattern.
- **Naming audit (executed in the design round against
  the real parser, and re-executed at build time):**
  census via `cli.build_parser()` walked recursively =
  82 distinct long options in the design round, 83 at
  build time (the tree gained one; see the census note
  after A9); `--compatibility-form`, `--compat-form`,
  `--compatibility-forms`, `--nfkc-foldables`,
  `--compat-foldables`, and `--foldables` are all
  **FREE**; `allow_abbrev=False` on all parsers walked.
  Shipped probe flags name the axis population in
  hyphenated noun form (`--canonical-form`,
  `--invisible-chars`, `--confusable-skeleton`).
  **Chosen: `--compatibility-form`** — it names the
  equivalence half being measured, in the shipped
  register, and it pairs exactly with
  `--canonical-form`: the two flags together say the two
  halves of UAX #15 §1.1, which is the §1 utility in one
  glance. It avoids `--nfkc-foldables` (names the
  *remedy* — a normalizer's operation — not the
  population; and SentencePiece's partial variant makes
  the acronym a claim the design does not make),
  `--compat-form` / `--foldables`
  (abbreviation/nickname register no shipped flag uses),
  and `--compatibility-forms` (plural-for-plural's-sake;
  the sibling is singular). USAGE section: **§8.62**,
  following §8.61 confusable skeleton. Module:
  `model_protocol_lab/compatibilityform.py`; group
  `compatibility_form`; stage details as above.

## 11. Duplication check (verified against the tree)

(A9) Read from `docs/USAGE.md` §8.1–§8.61 (61 probes):
no shipped *classified instrument over a non-`<wide>`
population* renders compatibility-equivalent forms
against their readings. Nearest neighbours, and why they
are not this:

- `canonical-form` (§8.54) is the sibling, disjoint by
  executed boundary: every member of this population
  satisfies NFC == NFD == ch, so §8.54's byte-divergence
  assert (`NFC(content) != NFD(content)`) excludes every
  case this probe generates, and its design defers the K
  leg explicitly ("No NFKC arm in v1") — this probe is
  that deferred leg, built as a separate instrument so
  neither headline pools the two equivalence halves.
- `perturbation` (§8.15) owns the `<wide>` fullwidth
  forms via its `unicode_variant` operator — excluded
  here by the general eligibility rule (104 `<wide>`
  rows, 0 admitted, executed); and perturbation blends
  the operator into an informational noise rate with no
  paired classification and no ids gate.
- Template surface-noise robustness (§8.31,
  `surfacenoise.py`) applies perturbation's
  `unicode_variant` operator (the `<wide>` class) as
  blended surface noise over template renders — no
  per-class paired classification, no ids gate, and its
  population is the one this design excludes by rule
  (A9).
- `numform` (§8.42) owns digit *grouping/surface form in
  answers*; this probe never asks for a number —
  fraction and superscript forms appear only as
  unreferenced carrier fields, and expected values stay
  in the word-number grammar (§9 R5).
- `invisible-chars` (§8.60) inserts Default_Ignorable
  code points (invisible, NFKC-stable as characters);
  disjoint population and mechanism (insertion vs
  substitution of visible foldables).
- `confusable-skeleton` (§8.61) substitutes
  skeleton-equivalent letters and *executes NFKC-identity
  for its whole population* as its boundary — the
  complement of this probe: every character there
  satisfies NFKC(ch) == ch; every character here
  satisfies NFKC(ch) != ch. The two populations are
  disjoint by that single decidable predicate.
- The enumeration candidate (wave-12 candidate 4) owns
  Roman numerals **as list markers** — its arms vary the
  itemizer/indexing glyph. This probe's Roman numerals
  are **content nouns** (a regnal name suffix,
  mid-sentence, §3 validator), never at an item start,
  never followed by `.` or `)` as a marker, and its
  question never refers to them. The validator rules,
  not the shared glyph family, are what keep the two
  instruments disjoint — stated here because a reviewer
  will check exactly this seam.
- Parked set (document-syntax, instruction-language,
  response-directive, multi-question) shares no axis
  with compatibility-form substitution;
  multi-question's contribution is its settlement
  obligation, discharged in §6 by construction.

Uniqueness is claimed for the **combination only** —
frozen UAX #15 compatibility-class substitution arms
(six classes, one field per class per case, uniform dose
1) × paired id-divergent classification with own-null
derived floors × completion-guarded labels ×
digest-committed treatments, on local checkpoints with
exact token sequences — never "first".

## 12. Rejected alternatives

- **Class-split panel (4 cases per class, 24 total):**
  the per-arm population caps at 4, below any honest
  MIN_D; the classification would be `insufficient` by
  construction. Rejected by arithmetic, not preference —
  it is the failure the §3 marker sentence exists to
  avoid.
- **Six subpanels of 24 (144 cases):** restores per-class
  panels at 6× the generation budget, to buy
  answer-bearing-word sites whose class coverage no
  natural word set provides uniformly anyway. The
  marker-sentence design buys the same per-arm N at the
  sibling budget; the cost (carrier position) is
  disclosed in §9 R3, not hidden.
- **One combined treated arm (all six fields treated at
  once):** dose 6, no per-class attribution — a fire
  would be unusable for §1's per-class ingest decision,
  and the family-wise and floors structure would
  describe a different (bundled) claim.
- **Fraction reading = raw NFKC (`1⁄2` with U+2044):**
  rejected in §9 R7 — the control must be the plain-text
  paste, which is ASCII; a control containing U+2044 is
  a normalizer's output, not a user's text.
- **Circled letters / superscript letters / SMALL ROMAN
  / non-digit `<super>` rows in v1:** §2's narrowing
  clauses — each broadens the population into another
  axis (letter fields, phonetic modifiers, enumeration
  markers) or breaks the uniform one-character marker
  field. Broadening is a new design, not a dose response
  (§9 R1 applies to population edits too).
- **U+FB05 (long-s ligature) via its two-stage NFKC:**
  its file target is not its reading, which would make
  it the one entry whose table cannot be read off the
  pinned file — the population-from-file discipline (§2)
  excludes it by the ASCII-target clause, stated in its
  staging row.
- **A winner readout:** §7 — not consumed by the
  utility; abstention-dominated at these rates.
- **An NFKC-normalized control arm** (render the control
  through NFKC and compare): that arm *is* the
  tokenizer-regime question the ids gate already answers
  without spending generations; and for the fraction
  class its text is the third form (§6), which would
  need its own reading rule — a different instrument.
- **User-fixture arm:** shipped fixtures are pure ASCII
  and a fixture arm would substitute into user data;
  generated-only is the sibling discipline
  (canonical-form invariant 3).
- **Two-directional canonical classification / borrowed
  floors / sign-test headline:** rejected by execution
  in the sibling rounds (§6); not re-litigated.

## 13. Build debts

1. The guard battery **with the ids dimension** in the
   build's own test suite (design-phase evidence is in
   the design round's scripts; the build re-executes
   worlds A–C under the pinned domain at the frozen cap,
   and its exhaustive audit states the reachable bound
   D + k ≤ 24).
2. Generator asserts with freeze tests: expected pure
   ASCII; control non-ASCII set == {the drawn kana
   reading}; the §3 enumeration screens (no newline; no
   sentence-initial Roman; `regnal Henry ` prefix); the
   §3 arithmetic screen (A7's operator set).
3. A **forced-collision freeze test** exercising the
   redraw path (slots 0+1 at counter+1, slots 2–9 at
   counter 0, the give-up counting A2 pins) — the
   design-phase seeds exercise it zero times (§3
   disclosure).
4. A named test that the shipped design doc's §2 tables,
   the module's tables, and a fresh derivation from the
   pinned UnicodeData.txt (by its SHA-256) all agree.
5. The build's OC tests reproduce **every** §6 figure
   from their own code, including the A1 M-conditioned
   rows and the exact paired values.
6. The decomposition battery of §5, including the
   precedence witnesses (exact expected answer;
   superscript echo under the ligature arm), A3's
   control-arm witness, and A5(iii)'s exhaustive
   echo-reachability search over all 216,000 expected
   values.

## 14. Sources

- UAX #15, Unicode Normalization Forms (unicode.org,
  **rev. 58 / Unicode 18.0.0**): §1.1 canonical vs
  compatibility equivalence (compatibility the weaker
  relation; Figure 2's circled/width/superscript/fraction
  subtypes, whose fraction schematisation `¼ → 1/4` uses
  the solidus — the §2 reading rule's primary support);
  §1.2 Table 1 (NFKC = compatibility decomposition +
  canonical composition) and the warning that the K
  forms "must not be blindly applied to arbitrary text"
  because they erase distinctions that may be important
  to the semantics; §1.2's statement that halfwidth and
  fullwidth katakana normalize to the same strings, "as
  will Roman numerals and their letter equivalents";
  §6's worked tables: U+FB03 (ffi ligature) is **not**
  decomposed in NFC/NFD and **is** decomposed in NFKC;
  "Henry Ⅳ" (U+2163) is unchanged in NFC/NFD and folds
  to "Henry IV" in NFKC. Both examples also executed
  against local data in the design round (NFKC(U+FB03)
  = `ffi`, NFC(U+FB03) = U+FB03; NFKC(`Henry Ⅳ`) =
  `Henry IV`).
- Pinned data file: `UnicodeData.txt` 15.0.0 from
  unicode.org/Public, SHA-256
  `806e9aed65037197f1ec85e12be6e8cd870fc5608b4de0fffd990f689f376a73`
  — every population number in §2 is derived from this
  file, cross-checked against the local interpreter's
  `unicodedata` (15.0.0).
- SentencePiece, "Text Normalization in SentencePiece"
  (SentencePiece documentation): "by default,
  SentencePiece normalizes input sentences using a custom
  variant of Unicode NFKC"; normalization "always happens
  *before* tokenization"; the default rule is `nmt_nfkc`;
  `identity` (no normalization) is an alternative; custom
  rules are "compiled and embedded directly into the
  model file"; and the caveat that SentencePiece "does
  not implement the entirety of Unicode NFKC
  normalization". Basis of the §6 regime
  pre-registration — an expectation about a *class* of
  pipelines, never an assertion about a specific
  checkpoint.
- Boucher, Shumailov, Anderson, Papernot, "Bad
  Characters: Imperceptible NLP Attacks",
  arXiv:2106.09898 — a single imperceptible encoding
  injection significantly reduces vulnerable systems'
  performance; three break most. **Supporting only**
  (§1): encoding-level perturbations are behaviourally
  real in deployed systems; no benign-task or
  NFKC-specific effect size is taken from it.
- Hua et al., "Flaw or Artifact? Rethinking Prompt
  Sensitivity in Evaluating LLMs", arXiv:2509.01790 —
  much reported prompt sensitivity stems from heuristic
  evaluation including rigid answer matching, across 7
  LLMs / 6 benchmarks / 12 templates. The §5
  decomposition obligation.
- Pecher, Spiegel, Belanec, Cegin, "Revisiting Prompt
  Sensitivity in Large Language Models for Text
  Classification: The Role of Prompt
  Underspecification", arXiv:2602.04297 — underspecified
  prompts show higher variance; instruction prompts
  less. The §9 R6 fully-specified-panel obligation.
- Tree: `core.py` (byte-exact `exact_match` conjunction);
  `density.py` (`sign_test_p`, two-sided — not used;
  `NEUTRAL_STATEMENTS`); `confidentiality.py`
  (`mcnemar_exact_p`, descriptive only); `perturbation.py`
  (`unicode_variant` fullwidth operator — the `<wide>`
  boundary); `surfacenoise.py` (§8.31 neighbour, A9);
  `canonicalform.py` (WORDS pool, invariants 2/9/10);
  `docs/CANONICALFORM_DESIGN.md` (the sibling instrument
  whose deferred K leg this is); `docs/USAGE.md`
  §8.1–§8.61 (duplication ground truth; §8.62 next free
  at design time, taken by this build); sibling designs
  and adjudications (guard settlement, amendment
  classes, the exact-paired and plateau disciplines, the
  0.8897 transplant figure).

**Removals:** none required — every source the design
relies on resolved when opened in the design round.
Corrections on contact with the data, recorded in place:
raw NFKC of a vulgar fraction is **not** the ASCII
reading shorthand implies (it carries U+2044; §2's
reading rule and §9 R7 carry the consequence); the
"superscripts" class is 249 `<super>` rows in the file,
of which the digit slice is 10 (§2 staging); and the
§2 staged counts carry their universes per A4.
