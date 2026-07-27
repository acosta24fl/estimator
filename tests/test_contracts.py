"""Futures symbology."""

import pandas as pd
import pytest

from mnq.data.contracts import (
    Contract,
    front_contract,
    generate_contracts,
    parse_contract,
    third_friday,
)


class TestThirdFriday:
    @pytest.mark.parametrize(
        "year,month,expected",
        [
            (2024, 3, "2024-03-15"),
            (2024, 6, "2024-06-21"),
            (2024, 9, "2024-09-20"),
            (2024, 12, "2024-12-20"),
            (2025, 3, "2025-03-21"),
            # A month starting on a Friday: the third Friday is the 15th.
            (2024, 11, "2024-11-15"),
            # A month starting on a Saturday pushes the first Friday to the 7th.
            (2025, 2, "2025-02-21"),
        ],
    )
    def test_known_expiries(self, year, month, expected):
        assert third_friday(year, month) == pd.Timestamp(expected)

    def test_always_a_friday(self):
        for year in range(2000, 2031):
            for month in (3, 6, 9, 12):
                assert third_friday(year, month).dayofweek == 4


class TestParseContract:
    def test_two_digit_year(self):
        c = parse_contract("NQZ24", reference=pd.Timestamp("2024-06-01"))
        assert (c.root, c.month, c.year) == ("NQ", 12, 2024)

    def test_four_digit_year(self):
        c = parse_contract("NQZ2024")
        assert (c.root, c.month, c.year) == ("NQ", 12, 2024)

    def test_single_digit_year_resolves_near_reference(self):
        c = parse_contract("NQZ4", reference=pd.Timestamp("2024-06-01"))
        assert c.year == 2024

    def test_single_digit_year_picks_the_closest_decade(self):
        c = parse_contract("NQZ9", reference=pd.Timestamp("2030-01-01"))
        assert c.year == 2029

    def test_micro_root(self):
        c = parse_contract("MNQH25", reference=pd.Timestamp("2025-01-01"))
        assert c.root == "MNQ" and c.month == 3

    def test_multiplier_ratio_matches_the_contract_specs(self):
        nq = parse_contract("NQZ24", reference=pd.Timestamp("2024-01-01"))
        mnq = parse_contract("MNQZ24", reference=pd.Timestamp("2024-01-01"))
        # This 10:1 ratio is why NQ history is usable for an MNQ model.
        assert nq.multiplier == 10 * mnq.multiplier

    @pytest.mark.parametrize("bad", ["", "NQ", "NQZ", "NQI24", "12345", "NQ24Z"])
    def test_rejects_nonsense(self, bad):
        with pytest.raises(ValueError):
            parse_contract(bad)

    def test_unlisted_roots_still_parse(self):
        """Deliberate: CL/GC/ZN must work without being in KNOWN_ROOTS."""
        c = parse_contract("CLZ24", reference=pd.Timestamp("2024-01-01"))
        assert (c.root, c.month, c.year) == ("CL", 12, 2024)
        assert c.multiplier == 1.0  # unknown root, no multiplier claimed

    def test_a_typo_that_fits_the_grammar_is_not_caught(self):
        """The documented cost of staying permissive about roots.

        ``NQZZ24`` reads as root 'NQZ'. Nothing in the code can tell that from
        a real product it has never heard of, so callers that need certainty
        filter on a known root instead.
        """
        assert parse_contract("NQZZ24", reference=pd.Timestamp("2024-01-01")).root == "NQZ"

    def test_roundtrips_through_code(self):
        c = Contract(year=2024, month=12, root="NQ")
        assert c.code == "NQZ24"
        assert c.code4 == "NQZ2024"
        assert parse_contract(c.code4) == c


class TestGenerateContracts:
    def test_quarterly_cycle_over_two_years(self):
        got = generate_contracts("NQ", "2023-01-01", "2024-12-31")
        assert [c.code for c in got] == [
            "NQH23", "NQM23", "NQU23", "NQZ23",
            "NQH24", "NQM24", "NQU24", "NQZ24",
        ]

    def test_expiry_ordered(self):
        got = generate_contracts("ES", "2020-01-01", "2025-12-31")
        assert got == sorted(got)
        assert all(
            a.expiry < b.expiry for a, b in zip(got, got[1:])
        )

    def test_excludes_expiries_outside_the_window(self):
        got = generate_contracts("NQ", "2024-04-01", "2024-10-01")
        assert [c.code for c in got] == ["NQM24", "NQU24"]


class TestFrontContract:
    def test_picks_the_nearest_unexpired(self):
        contracts = generate_contracts("NQ", "2024-01-01", "2024-12-31")
        got = front_contract(contracts, pd.Timestamp("2024-05-01"))
        assert got.code == "NQM24"

    def test_offset_rolls_early(self):
        contracts = generate_contracts("NQ", "2024-01-01", "2024-12-31")
        # Ten days before the June expiry, a 15-day offset should have moved on.
        got = front_contract(
            contracts, pd.Timestamp("2024-06-11"), offset_days=15
        )
        assert got.code == "NQU24"

    def test_returns_none_past_the_last_expiry(self):
        contracts = generate_contracts("NQ", "2024-01-01", "2024-12-31")
        assert front_contract(contracts, pd.Timestamp("2030-01-01")) is None
