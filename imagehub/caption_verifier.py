"""Verify Ideogram 4 JSON caption format."""

import json
import re
from collections.abc import Sequence
from pathlib import Path

PathLike = str | Path

NON_ASCII_UNICODE_ESCAPE_RE = re.compile(
    r"\\u(?:00[89a-fA-F][0-9a-fA-F]|0[1-9a-fA-F][0-9a-fA-F]{2}|[1-9a-fA-F][0-9a-fA-F]{3})"
)


class CaptionVerifier:
    """Verify Ideogram 4 JSON caption format."""

    top_level_known_keys: frozenset[str] = frozenset(
        {
            "high_level_description",
            "style_description",
            "compositional_deconstruction",
        }
    )
    style_description_key_order_photo: Sequence[str] = (
        "aesthetics",
        "lighting",
        "photo",
        "medium",
        "color_palette",
    )
    style_description_key_order_non_photo: Sequence[str] = (
        "aesthetics",
        "lighting",
        "medium",
        "art_style",
        "color_palette",
    )
    compositional_deconstruction_key_order: Sequence[str] = ("background", "elements")
    element_key_order_obj: Sequence[str] = ("type", "bbox", "desc", "color_palette")
    element_key_order_text: Sequence[str] = (
        "type",
        "bbox",
        "text",
        "desc",
        "color_palette",
    )
    style_description_known_keys: frozenset[str] = frozenset(
        {"aesthetics", "lighting", "photo", "art_style", "medium", "color_palette"}
    )
    element_known_keys: frozenset[str] = frozenset(
        {"type", "bbox", "text", "desc", "color_palette"}
    )
    element_types: frozenset[str] = frozenset({"obj", "text"})
    bbox_min = 0
    bbox_max = 1000
    style_description_palette_max = 16
    element_palette_max = 5

    def verify(self, caption: dict) -> list[str]:
        """Verify a parsed caption dict and return warning strings."""
        warnings: list[str] = []
        if not isinstance(caption, dict):
            warnings.append(f"root: expected a JSON object (dict), got {type(caption).__name__}")
            return warnings
        self._check_unknown_keys(caption, self.top_level_known_keys, "root", warnings)
        if "high_level_description" in caption:
            self._verify_high_level_description(caption["high_level_description"], warnings)
        if "style_description" in caption:
            self._verify_style_description(caption["style_description"], warnings)
        if "compositional_deconstruction" in caption:
            self._verify_compositional_deconstruction(
                caption["compositional_deconstruction"], warnings
            )
        else:
            warnings.append("root: 'compositional_deconstruction' must exist")
        return warnings

    def verify_raw(self, raw_text: str) -> list[str]:
        """Verify a raw JSON string, including its non-ASCII escaping."""
        warnings = self.check_ensure_ascii_false(raw_text)
        try:
            caption = json.loads(raw_text)
        except json.JSONDecodeError as error:
            warnings.append(f"invalid JSON: {error}")
            return warnings
        return warnings + self.verify(caption)

    def verify_file(self, path: PathLike) -> list[str]:
        """Verify a JSON file on disk."""
        return self.verify_raw(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def check_ensure_ascii_false(cls, raw_text: str, max_examples: int = 3) -> list[str]:
        warnings: list[str] = []
        matches = NON_ASCII_UNICODE_ESCAPE_RE.findall(raw_text)
        if not matches:
            return warnings
        if any(ord(character) > 0x7F for character in raw_text):
            return warnings
        examples = ", ".join(sorted(set(matches))[:max_examples])
        extra = "" if len(set(matches)) <= max_examples else ", ..."
        warnings.append(
            f"raw text: found {len(matches)} non-ASCII unicode escape(s) "
            f"(e.g. {examples}{extra}) and no literal non-ASCII characters. "
            "This usually means the file was saved with ensure_ascii=True; "
            "please re-save with json.dump(..., ensure_ascii=False) so non-ASCII "
            "characters are stored as literal UTF-8. "
            "Note: this is a warning, not an error, because the escapes could "
            "have been written intentionally — we cannot tell from the file alone."
        )
        return warnings

    def _verify_high_level_description(self, description: object, warnings: list[str]) -> None:
        if not isinstance(description, str):
            warnings.append(
                f"high_level_description: expected a string, got {type(description).__name__}"
            )

    def _verify_style_description(self, description: object, warnings: list[str]) -> None:
        if not isinstance(description, dict):
            warnings.append("style_description: expected a dict")
            return
        self._check_unknown_keys(
            description, self.style_description_known_keys, "style_description", warnings
        )
        has_photo = "photo" in description
        has_art_style = "art_style" in description
        if has_photo and has_art_style:
            warnings.append(
                "style_description: contains both 'photo' and 'art_style'; expected exactly one"
            )
            return
        if not has_photo and not has_art_style:
            warnings.append(
                "style_description: expected one of 'photo' (for photo captions) "
                "or 'art_style' (for non-photo captions)"
            )
            return
        self._check_key_order(
            description,
            self._style_description_key_order(description),
            "style_description",
            warnings,
        )
        if "color_palette" in description:
            self._verify_color_palette(
                description["color_palette"],
                "style_description.color_palette",
                self.style_description_palette_max,
                warnings,
            )

    def _verify_compositional_deconstruction(
        self, deconstruction: object, warnings: list[str]
    ) -> None:
        if not isinstance(deconstruction, dict):
            warnings.append("compositional_deconstruction: expected a dict")
            return
        if "background" not in deconstruction:
            warnings.append("compositional_deconstruction: 'background' must exist")
            return
        if not isinstance(deconstruction["background"], str):
            warnings.append(
                "compositional_deconstruction.background: expected a string, "
                f"got {type(deconstruction['background']).__name__}"
            )
            return
        if "elements" not in deconstruction:
            warnings.append("compositional_deconstruction: 'elements' must exist")
            return
        self._check_key_order(
            deconstruction,
            self.compositional_deconstruction_key_order,
            "compositional_deconstruction",
            warnings,
        )
        elements = deconstruction["elements"]
        if not isinstance(elements, list):
            warnings.append("compositional_deconstruction.elements: expected a list")
            return
        for index, element in enumerate(elements):
            self._verify_element(index, element, warnings)

    def _verify_element(self, index: int, element: object, warnings: list[str]) -> None:
        if not isinstance(element, dict):
            warnings.append(f"elements[{index}]: expected a dict")
            return
        self._check_unknown_keys(element, self.element_known_keys, f"elements[{index}]", warnings)
        if "type" not in element:
            warnings.append(f"elements[{index}]: 'type' must exist")
            return
        if element.get("type") not in self.element_types:
            warnings.append(f"elements[{index}]: 'type' must be one of {self.element_types}")
            return
        self._check_key_order(
            element, self._element_key_order(element), f"elements[{index}]", warnings
        )
        if "bbox" in element:
            self._verify_bbox(index, element["bbox"], warnings)
        if "color_palette" in element:
            self._verify_color_palette(
                element["color_palette"],
                f"elements[{index}].color_palette",
                self.element_palette_max,
                warnings,
            )

    def _verify_color_palette(
        self, palette: object, path: str, max_colors: int, warnings: list[str]
    ) -> None:
        if not isinstance(palette, list):
            warnings.append(f"{path}: expected a list")
            return
        if len(palette) > max_colors:
            warnings.append(
                f"{path}: too many colors ({len(palette)}), expected at most {max_colors}"
            )
            return
        for index, color in enumerate(palette):
            if (
                not isinstance(color, str)
                or len(color) != 7
                or color[0] != "#"
                or not all(character in "0123456789ABCDEF" for character in color[1:])
            ):
                warnings.append(f"{path}[{index}]: '{color}' is not a valid #RRGGBB hex color")

    def _verify_bbox(self, index: int, bbox: object, warnings: list[str]) -> None:
        if not isinstance(bbox, list) or len(bbox) != 4:
            warnings.append(f"elements[{index}].bbox: expected [ymin, xmin, ymax, xmax]")
            return
        if not all(isinstance(value, int) for value in bbox):
            warnings.append(f"elements[{index}].bbox: all values must be int")
            return
        ymin, xmin, ymax, xmax = bbox
        if not all(self.bbox_min <= value <= self.bbox_max for value in bbox):
            warnings.append(
                f"elements[{index}].bbox: values must be in "
                f"[{self.bbox_min}, {self.bbox_max}], got {bbox}"
            )
        if ymin > ymax:
            warnings.append(f"elements[{index}].bbox: ymin ({ymin}) > ymax ({ymax})")
        if xmin > xmax:
            warnings.append(f"elements[{index}].bbox: xmin ({xmin}) > xmax ({xmax})")

    def _element_key_order(self, element: dict) -> Sequence[str]:
        element_type = element.get("type")
        if element_type == "text":
            key_order = self.element_key_order_text
        elif element_type == "obj":
            key_order = self.element_key_order_obj
        else:
            raise ValueError
        return tuple(
            key for key in key_order if key not in {"bbox", "color_palette"} or key in element
        )

    def _style_description_key_order(self, description: dict) -> Sequence[str]:
        has_photo = "photo" in description
        has_art_style = "art_style" in description
        if has_art_style and not has_photo:
            key_order = self.style_description_key_order_non_photo
        elif not has_art_style and has_photo:
            key_order = self.style_description_key_order_photo
        else:
            raise ValueError
        return tuple(key for key in key_order if key != "color_palette" or key in description)

    @staticmethod
    def _check_key_order(
        obj: dict, expected_order: Sequence[str], path: str, warnings: list[str]
    ) -> None:
        present_keys = tuple(key for key in obj if key in expected_order)
        if present_keys != tuple(expected_order):
            warnings.append(f"{path}: key order is {present_keys}, expected {expected_order}")
        extra = [key for key in obj if key not in expected_order]
        if extra:
            warnings.append(f"{path}: keys {extra} are not allowed in this context")

    @staticmethod
    def _check_unknown_keys(
        obj: dict, known: frozenset[str], path: str, warnings: list[str]
    ) -> None:
        unknown = [key for key in obj if key not in known]
        if unknown:
            warnings.append(f"{path}: unknown keys {unknown} (not in schema)")
