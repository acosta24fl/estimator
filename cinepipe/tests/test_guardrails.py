import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cinepipe import castkit, scenecraft, shotcaller
from cinepipe.llm import Engine, EngineError
from cinepipe.models import CastkitRegistry, CharacterCastkit, ScenePlate, ScenecraftPlates, ShotList, StoryBible
from cinepipe.state import GateError, Workspace

EX = Path(__file__).resolve().parent.parent / "examples" / "handyman"


def load(name):
    return json.loads((EX / name).read_text())


@pytest.fixture
def bible():
    return StoryBible.model_validate(load("story_bible.json"))


@pytest.fixture
def cast():
    return CastkitRegistry.model_validate(load("castkit_registry.json"))


@pytest.fixture
def plates():
    return ScenecraftPlates.model_validate(load("scenecraft_plates.json"))


@pytest.fixture
def shots_data():
    return load("shotcaller_shots.json")


# ---------------- example passes every guardrail
def test_example_is_clean(bible, cast, plates, shots_data):
    shots = ShotList.model_validate(shots_data)
    assert castkit.check(bible, cast) == []
    assert scenecraft.check(bible, plates) == []
    assert shotcaller.check(shots, cast, plates) == []


# ---------------- Castkit: 18% gray, panels, face lock
def castkit_entry(**over):
    base = load("castkit_registry.json")["characters"][0]
    base.update(over)
    return base


@pytest.mark.parametrize("spec", [
    "flat neutral gray background, diffuse light",                       # no 18% / hex
    "18% gray (#808080) background",                                     # not 'neutral gray', not diffuse
    "flat 18% neutral gray background (#808080), diffuse studio lighting, dramatic rim light",  # directional
])
def test_gray_rule_rejects(spec):
    with pytest.raises(ValidationError):
        CharacterCastkit.model_validate(castkit_entry(background_lighting_spec=spec))


def test_gray_default_passes():
    assert "#808080" in CharacterCastkit.model_validate(castkit_entry()).background_lighting_spec


def test_face_lock_rejects_clothing():
    with pytest.raises(ValidationError, match="clothing"):
        CharacterCastkit.model_validate(castkit_entry(craniofacial_lock="Square jaw, brown eyes, wearing a navy cap pulled low over thick brows, age 41."))


def test_panels_must_be_front_back_portrait():
    with pytest.raises(ValidationError):
        CharacterCastkit.model_validate(castkit_entry(turnaround_panels=["Panel 1: Front Orthographic", "Panel 2: Side view", "Panel 3: Close-up"]))


def test_image_prompt_puts_face_before_wardrobe():
    p = CharacterCastkit.model_validate(castkit_entry()).image_prompt()
    assert p.index("FACE LOCK") < p.index("WARDROBE") < p.index("18% neutral gray")


# ---------------- Scenecraft: clean plate, optics
def plate_entry(**over):
    base = load("scenecraft_plates.json")["plates"][0]
    base.update(over)
    return base


@pytest.mark.parametrize("geo", [
    "Empty bathroom; a man fixes the sink.",
    "Tile wall, people in the hallway.",          # punctuation right after the word
    "Doorway with a silhouette of the customer.",
])
def test_clean_plate_bans_actors(geo):
    with pytest.raises(ValidationError, match="actor-free"):
        ScenePlate.model_validate(plate_entry(clean_plate_geometry=geo))


def test_clean_plate_allows_furniture_words():
    ScenePlate.model_validate(plate_entry(clean_plate_geometry="Free-standing vanity, sitting area by the window, empty room."))


@pytest.mark.parametrize("ap", ["f/1.2", "f/11", "2.8", "T2.8"])
def test_aperture_range(ap):
    with pytest.raises(ValidationError):
        ScenePlate.model_validate(plate_entry(aperture=ap))


def test_depth_cues_required():
    with pytest.raises(ValidationError, match="depth"):
        ScenePlate.model_validate(plate_entry(atmospheric_depth="crisp and sharp"))


# ---------------- Shotcaller: dialogue gate, beats, tier, isolation
def shot_with(shots_data, idx=3, **over):
    s = json.loads(json.dumps(shots_data["shots"][idx]))
    s.update(over)
    return s


def test_dialogue_crosstalk_rejected(shots_data):
    s = shot_with(shots_data)
    s["slot_14_serialized_dialogue_gate"][1]["start_sec"] = 2.0   # overlaps 0.4-2.6
    with pytest.raises(ValidationError, match="CROSSTALK"):
        ShotList.model_validate({"shots": [s]})


def test_listener_must_be_silent(shots_data):
    s = shot_with(shots_data)
    s["slot_14_serialized_dialogue_gate"][0]["silent_listeners_state"] = "@Homeowner reacts"
    with pytest.raises(ValidationError, match="mouths closed"):
        ShotList.model_validate({"shots": [s]})


def test_dialogue_past_duration_rejected(shots_data):
    s = shot_with(shots_data)
    s["slot_14_serialized_dialogue_gate"][1]["end_sec"] = 6.5
    with pytest.raises(ValidationError, match="runs past"):
        ShotList.model_validate({"shots": [s]})


def test_unbound_speaker_rejected(shots_data):
    s = shot_with(shots_data, slot_03_asset_bindings=["@Handyman", "@Bathroom_Plate"])
    with pytest.raises(ValidationError, match="isn't bound"):
        ShotList.model_validate({"shots": [s]})


def test_beats_need_timecodes(shots_data):
    s = shot_with(shots_data, slot_12_timecoded_action_beats=["He holds out the phone."])
    with pytest.raises(ValidationError, match="timecode"):
        ShotList.model_validate({"shots": [s]})


def test_1080p_locked_until_preview_approved(shots_data):
    s = shot_with(shots_data, slot_02_render_tier="1080p_master_render")
    with pytest.raises(ValidationError, match="1080p is locked"):
        ShotList.model_validate({"shots": [s]})
    s["approved_preview_seed"] = s["seed"]
    ShotList.model_validate({"shots": [s]})


def test_isolation_catches_repeated_phrase(shots_data, cast, plates):
    s = shot_with(shots_data, slot_11_atmosphere_volumetrics="Dust motes catch the window light, cracked tile visible between their shoulders.")
    errs = shotcaller.check(ShotList.model_validate({"shots": [s]}), cast, plates)
    assert any("appears in both" in e for e in errs)


def test_isolation_catches_wardrobe_leak(shots_data, cast, plates):
    s = shot_with(shots_data, slot_08_character_blocking="@Handyman in his navy cotton work shirt turns the phone toward her.")
    errs = shotcaller.check(ShotList.model_validate({"shots": [s]}), cast, plates)
    assert any("wardrobe of @Handyman leaks" in e for e in errs)


def test_slot09_must_reference_castkit_not_retype(shots_data, cast, plates):
    s = shot_with(shots_data, slot_09_wardrobe_lock="@Handyman: blue shirt and jeans. @Homeowner: castkit wardrobe_default.")
    errs = shotcaller.check(ShotList.model_validate({"shots": [s]}), cast, plates)
    assert any("never retype clothing" in e for e in errs)


def test_wardrobe_resolves_from_castkit(shots_data, cast):
    shot = ShotList.model_validate({"shots": [shot_with(shots_data)]}).shots[0]
    text = shotcaller.resolve_wardrobe(shot, cast)
    assert cast.get("@Handyman").wardrobe_default in text and "castkit" not in text


# ---------------- Stage gating
def test_gating_and_staleness(tmp_path, bible, cast, plates):
    ws = Workspace(tmp_path, "p")
    ws.create()
    with pytest.raises(GateError, match="has not been run"):
        ws.save("castkit", cast)
    ws.save("story_bible", bible)
    ws.save("castkit", cast)
    ws.save("scenecraft", plates)
    assert dict(ws.status())["castkit"] == "ok"
    edited = bible.model_copy(update={"title": "Edited"})
    ws.save("story_bible", edited)
    with pytest.raises(GateError, match="stale"):
        ws.load("castkit")


def test_hand_edited_artifact_rejected(tmp_path, bible):
    ws = Workspace(tmp_path, "p")
    ws.create()
    ws.save("story_bible", bible)
    data = json.loads(ws.path("story_bible").read_text())
    data["logline"] = "One. Two."
    ws.path("story_bible").write_text(json.dumps(data))
    with pytest.raises(GateError, match="fails its schema"):
        ws.load("story_bible")


# ---------------- Engine repair loop (fake client, no network)
class FakeMessages:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def parse(self, **kw):
        self.calls.append(kw)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        return SimpleNamespace(stop_reason="end_turn", stop_details=None, usage=usage, parsed_output=out)


def fake_engine(outputs):
    e = Engine.__new__(Engine)
    e.model, e.effort, e.max_repairs = "claude-opus-5-5", "high", 2
    e.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    e.client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(outputs)))
    return e


def test_engine_repairs_after_guardrail_failure(bible, cast):
    bad = CastkitRegistry(characters=cast.characters[:1])  # missing @Homeowner
    eng = fake_engine([bad, cast])
    assert castkit.build(eng, bible) == cast
    calls = eng.client.beta.messages.calls
    assert len(calls) == 2 and "Missing castkit entries" in calls[1]["messages"][-1]["content"]
    sys = calls[0]["system"]
    assert sys[-1]["cache_control"]["type"] == "ephemeral" and "STORY BIBLE" in sys[-1]["text"]
    assert calls[0]["model"] == "claude-opus-5-5" and calls[0]["fallbacks"] == "default"


def test_engine_gives_up_after_repairs(bible, cast):
    bad = CastkitRegistry(characters=cast.characters[:1])
    with pytest.raises(EngineError, match="kept failing"):
        castkit.build(fake_engine([bad, bad, bad]), bible)
