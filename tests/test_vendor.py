"""Vendor ingestion and the Parquet archive."""

import numpy as np
import pandas as pd
import pytest

from mnq.data.archive import Archive
from mnq.data.roll import build_continuous
from mnq.data.vendor import (
    DATABENTO,
    FIRSTRATE,
    GENERIC,
    VendorSpec,
    get_spec,
    load_contract_bars,
    normalise,
    read_file,
    validate_bars,
)


def _bars(n=48, start="2024-01-02 00:00", price=15_000.0):
    idx = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 5,
            "low": price - 5,
            "close": price,
            "volume": 1_000.0,
        },
        index=idx,
    )


class TestNormalise:
    def test_databento_scales_fixed_point_prices_and_ns_epochs(self):
        raw = pd.DataFrame(
            {
                "ts_event": [1_704_153_600_000_000_000, 1_704_157_200_000_000_000],
                "open": [15_000_000_000_000, 15_010_000_000_000],
                "high": [15_020_000_000_000, 15_030_000_000_000],
                "low": [14_990_000_000_000, 15_000_000_000_000],
                "close": [15_010_000_000_000, 15_020_000_000_000],
                "volume": [1_000, 1_200],
            }
        )
        got = normalise(raw, DATABENTO)
        assert got["close"].tolist() == [15_010.0, 15_020.0]
        assert str(got.index[0]) == "2024-01-02 00:00:00+00:00"
        assert got.index.tz is not None

    def test_eastern_timestamps_convert_to_utc(self):
        raw = _bars(n=2, start="2024-01-02 09:30").reset_index(names="timestamp")
        got = normalise(raw, FIRSTRATE)
        # 09:30 ET in January is 14:30 UTC.
        assert str(got.index[0]) == "2024-01-02 14:30:00+00:00"

    def test_column_renaming(self):
        spec = VendorSpec(
            name="acme",
            timestamp_column="date_time",
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"},
        )
        raw = pd.DataFrame(
            {
                "date_time": ["2024-01-02 00:00", "2024-01-02 01:00"],
                "o": [1.0, 2.0], "h": [3.0, 4.0],
                "l": [0.5, 1.5], "c": [2.0, 3.0], "v": [10, 20],
            }
        )
        got = normalise(raw, spec)
        assert list(got.columns) == ["open", "high", "low", "close", "volume"]

    def test_keeps_open_interest_when_present(self):
        raw = _bars(n=3).reset_index(names="timestamp")
        raw["open_interest"] = 500.0
        got = normalise(raw, GENERIC)
        assert "open_interest" in got.columns

    def test_drops_duplicate_timestamps_keeping_the_last(self):
        raw = _bars(n=3).reset_index(names="timestamp")
        raw = pd.concat([raw, raw.tail(1).assign(close=99_999.0)])
        got = normalise(raw, GENERIC)
        assert len(got) == 3
        assert got["close"].iloc[-1] == 99_999.0

    def test_drops_nonpositive_prices(self):
        raw = _bars(n=4).reset_index(names="timestamp")
        raw.loc[1, "close"] = 0.0
        assert len(normalise(raw, GENERIC)) == 3

    def test_missing_columns_are_an_error(self):
        raw = _bars(n=2).reset_index(names="timestamp").drop(columns=["volume"])
        with pytest.raises(ValueError, match="missing columns"):
            normalise(raw, GENERIC)

    def test_missing_timestamp_column_is_an_error(self):
        raw = _bars(n=2).reset_index(names="when")
        with pytest.raises(ValueError, match="timestamp column"):
            normalise(raw, GENERIC)

    def test_empty_input_gives_empty_output(self):
        assert normalise(pd.DataFrame(), GENERIC).empty


class TestValidateBars:
    def test_clean_bars_have_no_issues(self):
        assert validate_bars(normalise(_bars().reset_index(names="timestamp"), GENERIC)) == []

    def test_detects_inverted_range(self):
        df = normalise(_bars(n=5).reset_index(names="timestamp"), GENERIC)
        df.iloc[2, df.columns.get_loc("high")] = 1.0
        issues = validate_bars(df)
        assert any("high < low" in i for i in issues)

    def test_detects_body_outside_range(self):
        df = normalise(_bars(n=5).reset_index(names="timestamp"), GENERIC)
        df.iloc[1, df.columns.get_loc("close")] = 99_999.0
        assert any("outside the high-low range" in i for i in validate_bars(df))

    def test_detects_negative_volume(self):
        df = normalise(_bars(n=5).reset_index(names="timestamp"), GENERIC)
        df.iloc[0, df.columns.get_loc("volume")] = -1.0
        assert any("negative volume" in i for i in validate_bars(df))

    def test_reports_empty(self):
        assert validate_bars(pd.DataFrame(), "x") == ["x: empty"]


class TestReadFile:
    def test_splits_a_multi_contract_file_by_symbol(self, tmp_path):
        frames = []
        for code, price in (("NQZ24", 15_000.0), ("NQH25", 15_030.0)):
            f = _bars(n=5, price=price).reset_index(names="ts_event")
            f["symbol"] = code
            frames.append(f)
        path = tmp_path / "nq.csv"
        pd.concat(frames).to_csv(path, index=False)

        got = read_file(path, VendorSpec(name="t", timestamp_column="ts_event",
                                         symbol_column="symbol"))
        assert set(got) == {"NQZ24", "NQH25"}
        assert got["NQH25"]["close"].iloc[0] == 15_030.0

    def test_takes_the_contract_from_the_filename(self, tmp_path):
        path = tmp_path / "NQZ24_1min.csv"
        _bars(n=5).reset_index(names="timestamp").to_csv(path, index=False)
        got = read_file(path, FIRSTRATE)
        assert list(got) == ["NQZ24"]

    def test_unidentifiable_file_is_an_error(self, tmp_path):
        path = tmp_path / "mystery.csv"
        _bars(n=5).reset_index(names="timestamp").to_csv(path, index=False)
        with pytest.raises(ValueError, match="cannot tell which contract"):
            read_file(path, GENERIC)


class TestLoadContractBars:
    def _write(self, tmp_path, codes, prices=None):
        for i, code in enumerate(codes):
            price = prices[i] if prices else 15_000.0
            path = tmp_path / f"{code}_1min.csv"
            _bars(n=24, price=price).reset_index(names="timestamp").to_csv(
                path, index=False
            )

    def test_loads_every_contract(self, tmp_path):
        self._write(tmp_path, ["NQZ24", "NQH25", "NQM25"])
        got = load_contract_bars(tmp_path, FIRSTRATE, root="NQ")
        assert set(got) == {"NQZ24", "NQH25", "NQM25"}

    def test_filters_by_root(self, tmp_path):
        self._write(tmp_path, ["NQZ24", "ESZ24"])
        got = load_contract_bars(tmp_path, FIRSTRATE, root="NQ")
        assert set(got) == {"NQZ24"}

    def test_merges_files_for_the_same_contract(self, tmp_path):
        a = _bars(n=10, start="2024-01-02 00:00").reset_index(names="timestamp")
        b = _bars(n=10, start="2024-01-03 00:00").reset_index(names="timestamp")
        a.to_csv(tmp_path / "NQZ24_part1.csv", index=False)
        b.to_csv(tmp_path / "NQZ24_part2.csv", index=False)
        got = load_contract_bars(tmp_path, FIRSTRATE, root="NQ")
        assert len(got["NQZ24"]) == 20

    def test_a_corrupt_file_is_skipped_not_fatal(self, tmp_path):
        self._write(tmp_path, ["NQZ24"])
        (tmp_path / "NQH25_1min.csv").write_text("this is not a csv\n@@@@\n")
        got = load_contract_bars(tmp_path, FIRSTRATE, root="NQ")
        assert set(got) == {"NQZ24"}

    def test_strict_mode_raises_on_a_corrupt_file(self, tmp_path):
        self._write(tmp_path, ["NQZ24"])
        (tmp_path / "NQH25_1min.csv").write_text("nope\n@@@\n")
        with pytest.raises(Exception):
            load_contract_bars(tmp_path, FIRSTRATE, root="NQ", strict=True)

    def test_empty_directory_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_contract_bars(tmp_path, FIRSTRATE)

    def test_no_matching_root_is_an_error(self, tmp_path):
        self._write(tmp_path, ["ESZ24"])
        with pytest.raises(ValueError, match="no NQ contracts"):
            load_contract_bars(tmp_path, FIRSTRATE, root="NQ")


class TestSpecRegistry:
    def test_known_specs(self):
        for name in ("databento", "databento_pretty", "firstrate", "generic"):
            assert get_spec(name).name == name

    def test_unknown_spec_lists_the_options(self):
        with pytest.raises(ValueError, match="known:"):
            get_spec("bloomberg")


class TestArchive:
    def _contracts(self):
        out = {}
        for i, code in enumerate(["NQU24", "NQZ24"]):
            f = normalise(
                _bars(n=48, price=15_000.0 + 20 * i).reset_index(names="timestamp"),
                GENERIC,
            )
            out[code] = f
        return out

    def test_round_trips_contracts(self, tmp_path):
        archive = Archive(tmp_path)
        original = self._contracts()
        archive.write_contracts("NQ", "1h", original)
        got = archive.read_contracts("NQ", "1h")
        assert set(got) == set(original)
        pd.testing.assert_frame_equal(
            got["NQZ24"], original["NQZ24"], check_freq=False
        )

    def test_writing_twice_merges_rather_than_duplicates(self, tmp_path):
        archive = Archive(tmp_path)
        archive.write_contracts("NQ", "1h", self._contracts())
        archive.write_contracts("NQ", "1h", self._contracts())
        got = archive.read_contracts("NQ", "1h")
        assert len(got["NQZ24"]) == 48

    def test_lists_stored_contracts(self, tmp_path):
        archive = Archive(tmp_path)
        archive.write_contracts("NQ", "1h", self._contracts())
        assert archive.list_contracts("NQ", "1h") == ["NQU24", "NQZ24"]

    def test_reading_a_missing_product_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Archive(tmp_path).read_contracts("NQ", "1h")

    def test_round_trips_a_continuous_series(self, tmp_path):
        archive = Archive(tmp_path)
        idx = pd.date_range("2024-01-02", periods=200, freq="1h", tz="UTC")
        raw = {}
        for i, code in enumerate(["NQU24", "NQZ24"]):
            vol = np.where(idx < idx[100], 5_000.0 if i == 0 else 100.0,
                           100.0 if i == 0 else 5_000.0)
            price = pd.Series(15_000.0 + 20 * i, index=idx)
            raw[code] = pd.DataFrame(
                {"open": price, "high": price + 2, "low": price - 2,
                 "close": price, "volume": vol}
            )
        series = build_continuous(raw, method="volume")
        archive.write_continuous("NQ", "1h", series)

        got = archive.read_continuous("NQ", "1h")
        pd.testing.assert_frame_equal(got.bars, series.bars, check_freq=False)
        assert len(got.rolls) == len(series.rolls)
        assert got.rolls[0].gap == pytest.approx(series.rolls[0].gap)

    def test_describe_reports_what_is_stored(self, tmp_path):
        archive = Archive(tmp_path)
        archive.write_contracts("NQ", "1h", self._contracts())
        table = archive.describe()
        assert (table["kind"] == "contracts").any()
        assert table.loc[table["kind"] == "contracts", "items"].iloc[0] == 2
