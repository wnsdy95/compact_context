"""Tests for ConceptTable (v2 symbol table)."""

from ailang_ir.encoder.concept_table import ConceptTable, encode_id, decode_id


class TestBase36:
    def test_encode_zero(self):
        assert encode_id(0) == "0"

    def test_encode_small(self):
        assert encode_id(10) == "a"
        assert encode_id(35) == "z"

    def test_encode_two_digit(self):
        assert encode_id(36) == "10"
        assert encode_id(37) == "11"

    def test_round_trip(self):
        for n in [0, 1, 9, 10, 35, 36, 100, 999, 1295]:
            assert decode_id(encode_id(n)) == n

    def test_decode(self):
        assert decode_id("0") == 0
        assert decode_id("a") == 10
        assert decode_id("z") == 35
        assert decode_id("10") == 36


class TestConceptTable:
    def setup_method(self):
        self.ct = ConceptTable()

    def test_define_and_lookup(self):
        cid = self.ct.define("1to1_sent_map")
        assert cid == 0
        assert self.ct.lookup("1to1_sent_map") == 0

    def test_define_idempotent(self):
        id1 = self.ct.define("concept_a")
        id2 = self.ct.define("concept_a")
        assert id1 == id2
        assert self.ct.size == 1

    def test_sequential_ids(self):
        self.ct.define("a")
        self.ct.define("b")
        self.ct.define("c")
        assert self.ct.lookup("a") == 0
        assert self.ct.lookup("b") == 1
        assert self.ct.lookup("c") == 2

    def test_resolve(self):
        self.ct.define("my_concept")
        assert self.ct.resolve(0) == "my_concept"
        assert self.ct.resolve(99) is None

    def test_has(self):
        self.ct.define("x")
        assert self.ct.has("x") is True
        assert self.ct.has("y") is False

    def test_lookup_missing(self):
        assert self.ct.lookup("nonexistent") is None

    def test_size(self):
        assert self.ct.size == 0
        self.ct.define("a")
        assert self.ct.size == 1
        self.ct.define("b")
        assert self.ct.size == 2

    def test_ref_first_mention(self):
        ref = self.ct.ref("my_key")
        assert ref == "#my_key"
        assert self.ct.size == 1

    def test_ref_second_mention(self):
        self.ct.ref("my_key")
        ref2 = self.ct.ref("my_key")
        assert ref2 == "$0"

    def test_ref_multiple_concepts(self):
        r0 = self.ct.ref("concept_a")
        r1 = self.ct.ref("concept_b")
        r0_again = self.ct.ref("concept_a")
        r1_again = self.ct.ref("concept_b")
        assert r0 == "#concept_a"
        assert r1 == "#concept_b"
        assert r0_again == "$0"
        assert r1_again == "$1"


class TestConceptTablePersistence:
    def test_dump_and_restore(self):
        ct = ConceptTable()
        ct.define("alpha")
        ct.define("beta")
        ct.define("gamma")

        data = ct.dump()
        restored = ConceptTable.from_dump(data)

        assert restored.size == 3
        assert restored.lookup("alpha") == 0
        assert restored.lookup("beta") == 1
        assert restored.lookup("gamma") == 2
        assert restored.resolve(0) == "alpha"

    def test_dump_empty(self):
        ct = ConceptTable()
        data = ct.dump()
        restored = ConceptTable.from_dump(data)
        assert restored.size == 0

    def test_next_id_preserved(self):
        ct = ConceptTable()
        ct.define("a")
        ct.define("b")
        data = ct.dump()
        restored = ConceptTable.from_dump(data)
        new_id = restored.define("c")
        assert new_id == 2
