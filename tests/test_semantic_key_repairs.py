"""Semantic-key repairs: the seven defective frozen grading keys.

Spec citations: the frozen semantic answer-key audit
(``docs/audit/2026-11-10/`` program; lane report in
``agents_out/semantic_keys/report.md``; status rows in
``docs/STABLE_RELEASE_LEDGER.md`` section 2.3 and the SCI-B-01 row).
Each test pins one repaired key at every module occurrence: a key
whose fixture context does not uniquely entail its accepted answer
grades a factually defensible response as wrong, so the repairs pin
the entailment in the context (or, for D-1, replace the distractor)
without changing any spec's shape, schema or difficulty.
"""

import unittest

from model_protocol_lab import diffreports, fieldorder, position, structured, toolcall


def _structured_spec(spec_id):
    """Return the shipped structured spec with this id."""
    for spec in structured.STRUCTURED_SPECS:
        if spec["id"] == spec_id:
            return spec
    raise AssertionError(f"no structured spec {spec_id!r}")


def _mirror_spec(spec_id):
    """Return the diffreports field-order mirror spec with this id."""
    for spec in diffreports._FIELD_ORDER_SPECS:
        if spec["id"] == spec_id:
            return spec
    raise AssertionError(f"no mirror spec {spec_id!r}")


def _all_copies(spec_id):
    """Yield (origin, spec) for every graded copy of a shared spec."""
    yield "structured", _structured_spec(spec_id)
    yield "fieldorder", fieldorder.spec_by_id(spec_id)
    yield "diffreports mirror", _mirror_spec(spec_id)


class TestCapitalKey(unittest.TestCase):
    """D-1 / SCI-B-01: the position capital key accepted one of two
    European capitals (Lisbon and Oslo); the distractor is replaced
    with Cairo, a non-European capital of the same distractor class
    as Lima and Quito (Britannica: Lisbon is the capital of
    Portugal)."""

    def test_capital_options_and_key(self):
        """The capital spec's options and accepted answer are pinned."""
        spec = position.POSITION_SPECS[0]
        self.assertEqual(spec["id"], "capital")
        self.assertEqual(spec["options"], ("Cairo", "Oslo", "Lima", "Quito"))
        self.assertEqual(spec["correct"], "Oslo")

    def test_capital_exactly_one_european_capital(self):
        """Only the accepted option is a European capital."""
        spec = position.POSITION_SPECS[0]
        # Lisbon is a European capital; it must not appear as a
        # distractor. Every option remains a national capital, so
        # the distractor class (capitals of other continents) and
        # the item's difficulty are unchanged.
        self.assertNotIn("Lisbon", spec["options"])
        european_capitals = {"Oslo", "Lisbon", "Madrid", "Reykjavik"}
        self.assertEqual(set(spec["options"]) & european_capitals, {"Oslo"})


class TestExtractPersonKey(unittest.TestCase):
    """D-2: the context asserted Ada Lovelace was 36 when she wrote
    the first algorithm; she published the Notes in September 1843
    aged 27 (36 was her age at death). The fact is repaired in the
    context and the answer at every occurrence, including the
    canonicalisation worked example in the checking-rule texts."""

    def test_extract_person_age_27_everywhere(self):
        """Context and answer carry age 27 in all three spec copies."""
        for origin, spec in _all_copies("extract-person"):
            with self.subTest(origin=origin):
                self.assertEqual(
                    spec["context"],
                    "Ada Lovelace was 27 when she wrote the first algorithm.",
                )
                self.assertEqual(spec["answer"], {"name": "Ada Lovelace", "age": 27})

    def test_checking_rule_text_example_uses_27(self):
        """The canonicalisation worked example uses the repaired age."""
        for text in (fieldorder.CHECKING_RULE_TEXT, diffreports._FO_CHECKING_RULE_TEXT):
            self.assertIn('{"age": 27, "name": "Ada Lovelace"}', text)
            self.assertNotIn("'age': 36", text)


class TestRouteTicketKey(unittest.TestCase):
    """D-3: priority 3 was pinned by nothing (schema allows 1-5, the
    context carried no urgency cue); the ticket text now states its
    priority, making the value extraction rather than judgement."""

    def test_route_ticket_priority_stated_in_context(self):
        """The ticket carries '(priority 3 of 5)' in every copy."""
        for origin, spec in _all_copies("route-ticket"):
            with self.subTest(origin=origin):
                self.assertEqual(
                    spec["context"],
                    "Ticket (priority 3 of 5): 'My invoice shows a charge I do not recognize.'",
                )
                self.assertEqual(spec["answer"], {"department": "billing", "priority": 3})


class TestRouteAlertKey(unittest.TestCase):
    """D-4: channel pager was not entailed by a bare disk-usage
    alert (email and sms were equally defensible); the context now
    cues the channel explicitly."""

    def test_route_alert_channel_cued_in_context(self):
        """The alert text pages the on-call engineer in every copy."""
        for origin, spec in _all_copies("route-alert"):
            with self.subTest(origin=origin):
                self.assertEqual(
                    spec["context"],
                    "Alert: 'Disk usage on db-1 reached 97%. Page the on-call engineer.'",
                )
                self.assertEqual(spec["answer"], {"channel": "pager", "ack_required": True})


class TestRouteOrderKey(unittest.TestCase):
    """D-5: warehouse south pointed against the geography (Porto is
    in northern Portugal, Britannica) and express was unpinned by
    'by Friday'; the answer is north and the context states that
    express delivery was requested."""

    def test_route_order_north_and_express_pinned(self):
        """Warehouse is north and express is stated in every copy."""
        for origin, spec in _all_copies("route-order"):
            with self.subTest(origin=origin):
                self.assertEqual(
                    spec["context"],
                    "Order: 'Customer in Porto needs the parcel by Friday; "
                    "express delivery was requested.'",
                )
                self.assertEqual(spec["answer"], {"warehouse": "north", "express": True})


class TestRouteComplaintKey(unittest.TestCase):
    """D-6: team technical carried paired=True yet support was
    defensible (support received the complaint and confirmed the
    defect; nothing said it left support). The context now states
    the escalation, keeping the spec in the paired population."""

    def test_route_complaint_escalation_stated(self):
        """The context states the escalation to the technical team."""
        expected = (
            "Complaint: 'The update deleted my saved files.' "
            "Front-line support confirmed the update itself is defective "
            "and escalated the complaint to the technical team."
        )
        for origin, spec in (
            ("fieldorder", fieldorder.spec_by_id("route-complaint")),
            ("diffreports mirror", _mirror_spec("route-complaint")),
        ):
            with self.subTest(origin=origin):
                self.assertEqual(spec["context"], expected)
                self.assertEqual(
                    spec["answer"],
                    {"team": "technical", "defect_confirmed": True, "data_lost": True},
                )
                self.assertTrue(spec["paired"])


class TestDocsHealingKey(unittest.TestCase):
    """D-7: the expected call pinned top_k 3, an unstated default
    for an optional argument; the query now asks for the top 3
    results, mirroring the docs-refund sibling's phrasing."""

    def test_docs_healing_query_states_top_k(self):
        """The query states the result count its expected call pins."""
        query = next(q for q in toolcall.TOOLCALL_QUERIES if q["id"] == "docs-healing")
        self.assertEqual(
            query["question"],
            "Find documents about token healing; show the top 3 results.",
        )
        self.assertEqual(
            query["call"],
            {"name": "search_docs", "arguments": {"query": "token healing", "top_k": 3}},
        )


class TestMirrorConsistency(unittest.TestCase):
    """Consistency guards (pass before and after the repairs): the
    diff validator's mirror must stay byte-identical in content to
    the producer's frozen tables, or the mirror stops being an
    independent recomputation of the same instrument."""

    def test_diffreports_mirror_equals_fieldorder_table(self):
        """The mirror spec table equals the producer spec table."""
        self.assertEqual(diffreports._FIELD_ORDER_SPECS, fieldorder.FIELD_ORDER_SPECS)
        self.assertEqual(diffreports._FO_CHECKING_RULE_TEXT, fieldorder.CHECKING_RULE_TEXT)

    def test_shared_specs_agree_between_structured_and_fieldorder(self):
        """The nine shared specs carry identical contexts and answers."""
        for spec in structured.STRUCTURED_SPECS:
            with self.subTest(spec=spec["id"]):
                twin = fieldorder.spec_by_id(spec["id"])
                self.assertEqual(twin["context"], spec["context"])
                self.assertEqual(twin["answer"], spec["answer"])
                self.assertEqual(twin["question"], spec["question"])


if __name__ == "__main__":
    unittest.main()
