"""Tests for `inference.spec` — specifically the JSON-to-dataclass
resolver, including the enum-from-string coercion path that
`from __future__ import annotations` broke silently until a real
inference run caught it.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from osu.taiko2.inference.spec import build_config, load_spec


class _Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


# All three configs use `from __future__ import annotations` (implicit
# at module level — every test module in this repo does), so field
# types arrive as strings. build_config has to resolve them via
# typing.get_type_hints to coerce strings → enum members.

@dataclass(frozen=True, slots=True)
class _FlatConfig:
    __class__: "str" = ""  # never consulted; marker for resolver fixture
    name: str = ""
    count: int = 0
    color: _Color = _Color.RED


@dataclass(frozen=True, slots=True)
class _NestedConfig:
    name: str = ""
    inner: _FlatConfig | None = None


class TestBuildConfig:
    def test_enum_from_string_direct(self):
        node = {
            "__class__": f"{__name__}:_FlatConfig",
            "name": "a", "count": 3, "color": "GREEN",
        }
        cfg = build_config(node)
        assert cfg.color is _Color.GREEN  # not the string "GREEN"

    def test_enum_from_string_optional(self):
        # `color: _Color = ...` resolves cleanly; a default-None optional
        # field whose user-supplied value IS a string still coerces
        # correctly if the type is `Enum | None`.
        @dataclass(frozen=True, slots=True)
        class _OptColor:
            color: _Color | None = None

        node = {
            "__class__": f"{__name__}:_OptColor",
            "color": "BLUE",
        }
        # Can't use the module-level resolver since _OptColor is local
        # — build it by hand to exercise the Optional branch directly.
        from osu.taiko2.inference.spec import _coerce_field_value
        import typing
        hint = typing.get_type_hints(_OptColor)["color"]
        got = _coerce_field_value(hint, "BLUE")
        assert got is _Color.BLUE

    def test_nested_dataclass_via_class_marker(self):
        node = {
            "__class__": f"{__name__}:_NestedConfig",
            "name": "outer",
            "inner": {
                "__class__": f"{__name__}:_FlatConfig",
                "name": "inner", "count": 1, "color": "RED",
            },
        }
        cfg = build_config(node)
        assert cfg.name == "outer"
        assert cfg.inner is not None
        assert cfg.inner.color is _Color.RED

    def test_unknown_field_rejected(self):
        node = {
            "__class__": f"{__name__}:_FlatConfig",
            "color": "RED", "bogus_field": 42,
        }
        with pytest.raises(ValueError, match="unknown field"):
            build_config(node)

    def test_missing_class_marker_rejected(self):
        with pytest.raises(ValueError, match="missing __class__"):
            build_config({"name": "x"})


class TestLoadSpec:
    def test_rejects_both_sources(self):
        with pytest.raises(SystemExit):
            load_spec(config=Path("a.json"), config_json="{}")

    def test_rejects_neither_source(self):
        with pytest.raises(SystemExit):
            load_spec(config=None, config_json=None)

    def test_detects_missing_keys(self):
        with pytest.raises(SystemExit, match="missing keys"):
            load_spec(config=None, config_json='{"checkpoint": "x"}')
