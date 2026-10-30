"""Pydantic v2 schemas for every CinePipe stage.

The guardrails live here, in validators, so a stage can never hand a bad
artifact downstream no matter what the LLM returned:

- Castkit: 18% neutral gray (#808080), shadowless reference sheets with the
  exact 3-panel turnaround, and a face lock that never mentions clothing.
- Scenecraft: actor-free clean plates with real-world optics.
- Shotcaller: the 16-slot spine, timecoded beats, the
  ONE MOUTH SPEAKS AT A TIME dialogue gate, and the 480p-preview-first
  render tier.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TAG = r"^@[A-Za-z][A-Za-z0-9_]*$"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


# ==========================================================================
# STAGE 1: STORY BIBLE (6 narrative pillars)
# ==========================================================================
PILLARS = ("premise", "timeline", "factions", "world_rules", "locations", "characters")


class Era(Strict):
    timeline: list[str] = Field(..., min_length=1, description="Chronological events, oldest first")
    era_rules: list[str] = Field(..., min_length=1, description="What exists / doesn't exist in this era (tech, money, customs)")


class Faction(Strict):
    name: str
    goal: str
    power_over: str = Field(..., description="Who or what this faction has leverage over, and how")


class Location(Strict):
    loc_id: str = Field(..., pattern=TAG, description="Tag like @Bathroom")
    name: str
    visual_motifs: list[str] = Field(..., min_length=1)
    color_palette: list[str] = Field(..., min_length=2, description="Hex colors or named colors")
    architecture: str


class CharacterProfile(Strict):
    char_id: str = Field(..., pattern=TAG, description="Tag like @Handyman")
    name: str
    age: int = Field(..., ge=1, le=120)
    archetype: str
    flaw: str
    want: str = Field(..., description="What they want in this story")
    vocal_cadence: str = Field(..., description="How they talk: pace, vocabulary, verbal habits")
    behavioral_ticks: list[str] = Field(..., min_length=1)


class StoryBible(Strict):
    title: str
    premise: str
    logline: str = Field(..., description="One sentence: who wants what, what's in the way")
    era: Era
    factions: list[Faction] = Field(..., min_length=1)
    world_rules: list[str] = Field(..., min_length=1, description="Physics/tech/social rules the story can never break")
    locations: list[Location] = Field(..., min_length=1)
    characters: list[CharacterProfile] = Field(..., min_length=1)

    @field_validator("logline")
    @classmethod
    def one_sentence(cls, v: str) -> str:
        if len(re.findall(r"[.!?](\s|$)", v)) > 1:
            raise ValueError("Logline must be a single sentence.")
        return v

    @model_validator(mode="after")
    def unique_tags(self) -> "StoryBible":
        for kind, ids in (("character", [c.char_id for c in self.characters]), ("location", [l.loc_id for l in self.locations])):
            dupes = {i for i in ids if ids.count(i) > 1}
            if dupes:
                raise ValueError(f"Duplicate {kind} tags: {sorted(dupes)}")
        return self


class InterviewStep(Strict):
    """One turn of the Story Bible interview: either another question, or the finished bible."""
    locked_pillars: list[Literal["premise", "timeline", "factions", "world_rules", "locations", "characters"]]
    next_question: str | None = Field(None, description="Ask exactly one question about the weakest unlocked pillar; null when all 6 are locked")
    bible: StoryBible | None = Field(None, description="The compiled bible; only when all 6 pillars are locked")

    @model_validator(mode="after")
    def question_or_bible(self) -> "InterviewStep":
        if set(self.locked_pillars) == set(PILLARS):
            if self.bible is None:
                raise ValueError("All pillars are locked, so the compiled bible is required.")
        elif not self.next_question:
            raise ValueError("Pillars are still open, so next_question is required.")
        return self


# ==========================================================================
# STAGE 2: CASTKIT (face lock + 18% gray 3-panel turnaround)
# ==========================================================================
NEUTRAL_GRAY_SPEC = "flat 18% neutral gray background (#808080), even diffuse studio lighting, zero directional shadows"
REQUIRED_PANELS = (
    "Panel 1: Full-body Front Orthographic View, neutral A-pose",
    "Panel 2: Full-body Back Orthographic View, neutral A-pose",
    "Panel 3: Extreme Close-Up Neutral Expression Portrait, direct eye contact",
)
WARDROBE_WORDS = {
    "shirt", "t-shirt", "jacket", "coat", "hat", "cap", "hoodie", "sweater", "jeans", "pants", "trousers",
    "dress", "skirt", "boots", "shoes", "sneakers", "belt", "uniform", "vest", "scarf", "gloves", "apron",
    "overalls", "helmet", "glasses", "sunglasses", "tie", "suit", "sleeve", "sleeves", "collar",
}
DIRECTIONAL_LIGHT = ("rim light", "backlit", "backlight", "key light from", "golden hour", "dramatic lighting",
                     "hard shadow", "chiaroscuro", "spotlight", "sunlight", "moody", "low-key")


class WardrobeLayer(Strict):
    context: str = Field(..., description="Act or location this outfit belongs to, e.g. 'Act 1' or '@Bathroom'")
    outfit: str = Field(..., description="Fabric, cut, color for this context")


class CharacterCastkit(Strict):
    char_id: str = Field(..., pattern=TAG)
    craniofacial_lock: str = Field(..., min_length=40, description="Immutable face/head only: jawline, eye spacing, nose, skin texture, marks, hair, age. NO clothing.")
    body_lock: str = Field(..., description="Height, build, posture. NO clothing.")
    wardrobe_default: str
    wardrobe_layers: list[WardrobeLayer] = Field(default_factory=list)
    turnaround_panels: list[str] = Field(default_factory=lambda: list(REQUIRED_PANELS))
    background_lighting_spec: str = NEUTRAL_GRAY_SPEC
    reference_image_paths: list[str] = Field(default_factory=list, description="Filled in after you generate the sheet")

    @field_validator("craniofacial_lock", "body_lock")
    @classmethod
    def no_clothing_in_identity(cls, v: str) -> str:
        hits = _words(v) & WARDROBE_WORDS
        if hits:
            raise ValueError(f"Identity locks must not mention clothing (found {sorted(hits)}); put it in a wardrobe layer.")
        return v

    @field_validator("turnaround_panels")
    @classmethod
    def exact_three_panels(cls, v: list[str]) -> list[str]:
        if [p.split(":")[0].strip() for p in v] != ["Panel 1", "Panel 2", "Panel 3"]:
            raise ValueError("Turnaround must be exactly Panel 1 (front), Panel 2 (back), Panel 3 (ECU portrait).")
        need = ("front orthographic", "back orthographic", "close-up")
        for panel, word in zip(v, need):
            if word not in panel.lower():
                raise ValueError(f"'{panel}' must be a {word} view.")
        return v

    @field_validator("background_lighting_spec")
    @classmethod
    def enforce_neutral_gray(cls, v: str) -> str:
        low = v.lower()
        missing = [k for k in ("18%", "#808080", "neutral gray", "diffuse") if k not in low]
        if missing:
            raise ValueError(f"Reference sheets need an 18% neutral gray (#808080) diffuse setup; missing {missing}.")
        bad = [d for d in DIRECTIONAL_LIGHT if d in low]
        if bad:
            raise ValueError(f"Directional lighting would get baked into the reference: {bad}.")
        return v

    def image_prompt(self, wardrobe: str | None = None) -> str:
        """Face lock first, then body, then clothing, then the sheet + lighting spec."""
        return (
            f"Character reference sheet for {self.char_id}. "
            f"FACE LOCK: {self.craniofacial_lock} BODY: {self.body_lock} "
            f"WARDROBE: {wardrobe or self.wardrobe_default}. "
            f"Layout: three panels side by side - {'; '.join(self.turnaround_panels)}. "
            f"{self.background_lighting_spec}. Same person in all panels, photographic, no text, no logos."
        )


class CastkitRegistry(Strict):
    characters: list[CharacterCastkit] = Field(..., min_length=1)

    def get(self, char_id: str) -> CharacterCastkit:
        for c in self.characters:
            if c.char_id == char_id:
                return c
        raise KeyError(char_id)


# ==========================================================================
# STAGE 3: SCENECRAFT (actor-free clean plates + real optics)
# ==========================================================================
ACTOR_WORDS = {
    "person", "persons", "people", "man", "men", "woman", "women", "child", "children", "kid", "kids",
    "crowd", "crowds", "actor", "actors", "figure", "figures", "silhouette", "silhouettes", "pedestrian",
    "pedestrians", "character", "characters", "customer", "worker", "workers", "someone", "somebody",
    "walking", "talking", "he", "she", "his", "her", "him",
}
APERTURE = re.compile(r"^f/(\d+(?:\.\d+)?)$")


class ScenePlate(Strict):
    plate_id: str = Field(..., pattern=TAG, description="Tag like @Bathroom_Plate")
    location_ref: str = Field(..., pattern=TAG, description="@loc_id from the Story Bible")
    clean_plate_geometry: str = Field(..., description="Empty set: architecture, props, layout. No people of any kind.")
    focal_length_mm: Literal[18, 24, 35, 50, 85, 100, 135] = 35
    lens_type: Literal["Cooke Panchro (spherical)", "ARRI Master Anamorphic", "Zeiss Supreme Prime (spherical)"] = "Cooke Panchro (spherical)"
    aperture: str = "f/2.8"
    shutter_angle_deg: Literal[90, 180, 270, 360] = 180
    color_temp_k: int = Field(4300, ge=1800, le=10000)
    practical_lights: list[str] = Field(default_factory=list)
    atmospheric_depth: str = Field(..., description="Aerial perspective, micro-dust, halation, film grain")

    @field_validator("clean_plate_geometry")
    @classmethod
    def ban_actors(cls, v: str) -> str:
        hits = _words(v) & ACTOR_WORDS
        if hits:
            raise ValueError(f"Clean plate must be actor-free; found {sorted(hits)}.")
        return v

    @field_validator("aperture")
    @classmethod
    def real_aperture(cls, v: str) -> str:
        m = APERTURE.match(v)
        if not m or not 1.8 <= float(m.group(1)) <= 8:
            raise ValueError("Aperture must look like f/2.8 and sit between f/1.8 and f/8.")
        return v

    @field_validator("atmospheric_depth")
    @classmethod
    def has_depth_cues(cls, v: str) -> str:
        cues = ("aerial perspective", "dust", "halation", "halated", "grain", "haze")
        if sum(c in v.lower() for c in cues) < 2:
            raise ValueError(f"Atmospheric depth needs at least two depth cues from {cues}.")
        return v

    def optics(self) -> str:
        return (f"{self.focal_length_mm}mm {self.lens_type}, {self.aperture}, "
                f"{self.shutter_angle_deg}-degree shutter, {self.color_temp_k}K")

    def image_prompt(self) -> str:
        return (f"VFX clean plate, empty set, locked architectural geometry. {self.clean_plate_geometry} "
                f"Practicals: {', '.join(self.practical_lights) or 'none'}. Shot on {self.optics()}. "
                f"{self.atmospheric_depth}. No people, no figures, no silhouettes, no text.")


class ScenecraftPlates(Strict):
    plates: list[ScenePlate] = Field(..., min_length=1)

    def get(self, plate_id: str) -> ScenePlate:
        for p in self.plates:
            if p.plate_id == plate_id:
                return p
        raise KeyError(plate_id)


# ==========================================================================
# STAGE 4: SHOTCALLER (16-slot spine + dialogue gate)
# ==========================================================================
BEAT = re.compile(r"^\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)s\]\s+\S")


class ShotHeader(Strict):
    shot_id: str = Field(..., pattern=r"^SH\d{3}$")
    duration_s: float = Field(..., gt=0, le=15)
    fps: Literal[24, 25, 30] = 24

    def __str__(self) -> str:
        return f"{self.shot_id}, {self.duration_s:.1f}s, {self.fps}fps"


class DialogueSegment(Strict):
    start_sec: float = Field(..., ge=0)
    end_sec: float = Field(..., gt=0)
    active_speaker_id: str = Field(..., pattern=TAG)
    spoken_line: str = Field(..., min_length=1)
    silent_listeners_state: str = Field(..., description="Every other on-screen character, e.g. '@Homeowner mouth closed, nodding'")

    @model_validator(mode="after")
    def sane(self) -> "DialogueSegment":
        if self.end_sec <= self.start_sec:
            raise ValueError("Dialogue segment must end after it starts.")
        low = self.silent_listeners_state.lower()
        if "closed" not in low and "silent" not in low and "off-screen" not in low and "none" not in low:
            raise ValueError("silent_listeners_state must say listeners have mouths closed / are silent.")
        return self

    def __str__(self) -> str:
        return (f"[{self.start_sec:05.2f}-{self.end_sec:05.2f}] {self.active_speaker_id} speaks: "
                f"\"{self.spoken_line}\" (mouth moving) | {self.silent_listeners_state}")


class Shotcaller16SlotPrompt(Strict):
    slot_01_shot_id_duration: ShotHeader
    slot_02_render_tier: Literal["480p_preview_draft", "1080p_master_render"] = "480p_preview_draft"
    slot_03_asset_bindings: list[str] = Field(..., min_length=1)
    slot_04_shot_scale_framing: str
    slot_05_camera_optics: str
    slot_06_camera_rig_motion: str
    slot_07_geometry_spatial_matte: str
    slot_08_character_blocking: str
    slot_09_wardrobe_lock: str
    slot_10_lighting_key_ratio: str
    slot_11_atmosphere_volumetrics: str
    slot_12_timecoded_action_beats: list[str] = Field(..., min_length=1)
    slot_13_facial_micro_expressions: str
    slot_14_serialized_dialogue_gate: list[DialogueSegment] = Field(default_factory=list)
    slot_15_physics_interaction_sim: str
    slot_16_negative_constraints: str = (
        "no simultaneous lip movement, no morphing geometry, no baked directional shadows, "
        "no extra limbs or fingers, no camera jitter, no text or logos"
    )
    # render bookkeeping (not part of the prompt)
    seed: int = Field(..., ge=0)
    approved_preview_seed: int | None = None

    @field_validator("slot_03_asset_bindings")
    @classmethod
    def tags_only(cls, v: list[str]) -> list[str]:
        bad = [t for t in v if not re.match(TAG, t)]
        if bad:
            raise ValueError(f"Asset bindings must be @tags only, got {bad}.")
        if not any(t.endswith("_Plate") for t in v):
            raise ValueError("Every shot must bind exactly one @..._Plate clean plate.")
        return v

    @field_validator("slot_12_timecoded_action_beats")
    @classmethod
    def timecoded(cls, v: list[str]) -> list[str]:
        for b in v:
            if not BEAT.match(b):
                raise ValueError(f"Beat '{b}' must start with a timecode like [0-2s].")
        return v

    @field_validator("slot_14_serialized_dialogue_gate")
    @classmethod
    def one_mouth_speaks_at_a_time(cls, segs: list[DialogueSegment]) -> list[DialogueSegment]:
        segs = sorted(segs, key=lambda s: s.start_sec)
        for a, b in zip(segs, segs[1:]):
            if a.end_sec > b.start_sec:
                raise ValueError(
                    f"DIALOGUE CROSSTALK: {a.active_speaker_id} ({a.start_sec}-{a.end_sec}s) overlaps "
                    f"{b.active_speaker_id} ({b.start_sec}-{b.end_sec}s). ONE MOUTH SPEAKS AT A TIME.")
        return segs

    @model_validator(mode="after")
    def fits_duration_and_tier(self) -> "Shotcaller16SlotPrompt":
        dur = self.slot_01_shot_id_duration.duration_s
        for b in self.slot_12_timecoded_action_beats:
            end = float(BEAT.match(b).group(2))
            if end > dur + 1e-6:
                raise ValueError(f"Beat '{b}' runs past the {dur}s shot.")
        for s in self.slot_14_serialized_dialogue_gate:
            if s.end_sec > dur + 1e-6:
                raise ValueError(f"Dialogue by {s.active_speaker_id} runs past the {dur}s shot.")
            if s.active_speaker_id not in self.slot_03_asset_bindings:
                raise ValueError(f"{s.active_speaker_id} speaks but isn't bound in slot 03.")
        if self.slot_02_render_tier == "1080p_master_render" and self.approved_preview_seed != self.seed:
            raise ValueError("1080p is locked until the 480p preview with this seed is approved (cinepipe approve).")
        return self

    def slots(self, wardrobe: str | None = None) -> list[tuple[str, str]]:
        """`wardrobe` replaces slot 09 with the Castkit-resolved outfit text when compiling."""
        dlg = " || ".join(str(s) for s in self.slot_14_serialized_dialogue_gate) or "NO DIALOGUE - all mouths closed"
        return [
            ("SHOT_ID & DURATION", str(self.slot_01_shot_id_duration)),
            ("RENDER_TIER", f"{self.slot_02_render_tier} (seed {self.seed})"),
            ("ASSET_BINDINGS", " ".join(self.slot_03_asset_bindings)),
            ("SHOT_SCALE & FRAMING", self.slot_04_shot_scale_framing),
            ("CAMERA_OPTICS", self.slot_05_camera_optics),
            ("CAMERA_RIG & MOTION", self.slot_06_camera_rig_motion),
            ("GEOMETRY & SPATIAL MATTE", self.slot_07_geometry_spatial_matte),
            ("CHARACTER_BLOCKING", self.slot_08_character_blocking),
            ("WARDROBE_LOCK", wardrobe or self.slot_09_wardrobe_lock),
            ("LIGHTING_KEY & RATIO", self.slot_10_lighting_key_ratio),
            ("ATMOSPHERE & VOLUMETRICS", self.slot_11_atmosphere_volumetrics),
            ("TIMECODED_ACTION_BEATS", " ".join(self.slot_12_timecoded_action_beats)),
            ("FACIAL_MICRO_EXPRESSIONS", self.slot_13_facial_micro_expressions),
            ("SERIALIZED_DIALOGUE_GATE", "ONE MOUTH SPEAKS AT A TIME. " + dlg),
            ("PHYSICS & INTERACTION_SIM", self.slot_15_physics_interaction_sim),
            ("NEGATIVE_CONSTRAINTS", self.slot_16_negative_constraints),
        ]

    def compile(self, wardrobe: str | None = None) -> str:
        return "\n".join(f"[{i:02d} {name}] {text}" for i, (name, text) in enumerate(self.slots(wardrobe), 1))


class ShotList(Strict):
    shots: list[Shotcaller16SlotPrompt] = Field(..., min_length=1)


# ==========================================================================
# PIPELINE STATE
# ==========================================================================
class StageRecord(Strict):
    path: str
    sha256: str
    upstream_sha256: dict[str, str] = Field(default_factory=dict)


class ProjectState(Strict):
    name: str
    stages: dict[Literal["story_bible", "castkit", "scenecraft", "shotcaller"], StageRecord] = Field(default_factory=dict)
