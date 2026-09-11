"""Parse a URDF to extract the pose of every AprilTag link in the object's base frame.

Convention: a tag belonging to AprilTag family ``<family>`` with numeric id
``<id>`` is a child link named ``"apriltag<family>_<id>"`` rigidly fixed-jointed
to the object's base link (possibly through intermediate fixed-joint links such
as ``dock_point``). Examples::

    apriltag25h9_0   -> family="tag25h9",  id=0
    apriltag36h11_5  -> family="tag36h11", id=5

The literal ``"apriltag"`` prefix is configurable via ``prefix``; the
``<family>`` token matches any canonical apriltag family name (regex
``tag<digits>h<digits>``), and the ``_<id>`` suffix must be a non-negative
integer. The pose we want is the homogeneous transform from the base frame
to the tag frame.

The numeric id is *per-family*: tag25h9 and tag36h11 may each number their
own tags starting from 0 without colliding.

We use a tiny stdlib-only XML walker instead of pulling in
``urdf_parser_py``; the only fields we need are ``<joint>``'s ``child`` link
name and its optional ``<origin xyz rpy/>`` child.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R


# After stripping the literal "apriltag" prefix from a child link name,
# the remainder is "<family_short>_<id>", e.g. "25h9_0" or "36h11_5".
# The "tag" prefix (e.g. "tag25h9") is intentionally NOT repeated in the
# link name; it is re-added in `_parse_tag_link` so the returned family
# matches the canonical naming ("tag25h9", "tag36h11", ...).
_TAG_LINK_RE = re.compile(r"^(?P<family_short>[A-Za-z0-9]+)_(?P<id>\d+)$")


def _parse_floats(text: Optional[str], n: int) -> list[float]:
    if not text:
        return [0.0] * n
    parts = text.replace(",", " ").split()
    vals = [float(x) for x in parts]
    if len(vals) < n:
        vals += [0.0] * (n - len(vals))
    return vals[:n]


def _origin_to_homogeneous(origin: Optional[ET.Element]) -> np.ndarray:
    """Parse a URDF ``<origin xyz="..." rpy="..."/>`` element into a 4x4 transform.

    Missing / empty fields default to zeros (xyz) / identity (rpy).
    """
    xyz = _parse_floats(origin.get("xyz") if origin is not None else None, 3)
    rpy = _parse_floats(origin.get("rpy") if origin is not None else None, 3)
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
    T[:3, 3] = xyz
    return T


@dataclass
class TagTransform:
    """A single AprilTag's rigid pose in the object's base frame."""
    tag_id: int
    family: str
    child_link: str
    T_base_to_tag: np.ndarray  # 4x4


def _load_link_names(urdf_path: Path) -> Set[str]:
    """Return the set of all ``<link name="...">`` names in the URDF."""
    root = ET.parse(str(urdf_path)).getroot()
    return {lnk.get("name") for lnk in root.findall("link") if lnk.get("name")}


def _resolve_base_link(urdf_path: Path, base_link_name: Optional[str]) -> str:
    """Pick the URDF root link (or validate an explicit override).

    A "root" link is any link that no other joint's ``<child>`` references.
    If multiple roots exist the first one in document order is returned.
    """
    link_names = _load_link_names(urdf_path)
    if base_link_name:
        if base_link_name not in link_names:
            raise ValueError(
                f"base_link_name '{base_link_name}' not in URDF link set"
            )
        return base_link_name
    root = ET.parse(str(urdf_path)).getroot()
    child_links = set()
    for j in root.findall("joint"):
        ch = j.find("child")
        if ch is not None and ch.get("link"):
            child_links.add(ch.get("link"))
    roots = [lnk.get("name") for lnk in root.findall("link") if lnk.get("name")]
    for name in roots:
        if name not in child_links:
            return name
    if roots:
        return roots[0]
    raise ValueError(f"No <link> elements found in {urdf_path}")


def _parse_tag_link(child_link: str, prefix: str) -> Optional[Tuple[str, int]]:
    """Extract ``(family, tag_id)`` from a child link name, or ``None``.

    The link must start with ``prefix`` (typically ``"apriltag"``) followed
    by ``<family_short>_<id>``. ``<family_short>`` is the apriltag family
    name without the literal ``"tag"`` prefix (e.g. ``"25h9"``); the
    function re-adds it so the returned family matches the canonical
    name (``"tag25h9"``). ``<id>`` is a non-negative int.

    Examples (with ``prefix="apriltag"``)::
        apriltag25h9_0   -> ("tag25h9",  0)
        apriltag36h11_5  -> ("tag36h11", 5)
    """
    if not child_link.startswith(prefix):
        return None
    suffix = child_link[len(prefix):]
    m = _TAG_LINK_RE.match(suffix)
    if m is None:
        return None
    return f"tag{m.group('family_short')}", int(m.group("id"))


def _build_fixed_joint_parent_map(
    urdf_path: Path,
) -> Dict[str, Tuple[str, np.ndarray]]:
    """Return ``{child_link: (parent_link, T_parent_to_child)}`` for every fixed joint.

    Joints whose ``<child>`` references a link already populated by an
    earlier fixed joint raise ``ValueError`` (URDF has exactly one parent
    per link). Non-fixed joints are skipped; the caller is responsible for
    detecting non-fixed chains during a walk.
    """
    root = ET.parse(str(urdf_path)).getroot()
    parent_of: Dict[str, Tuple[str, np.ndarray]] = {}
    for joint in root.findall("joint"):
        if joint.get("type", "").lower() != "fixed":
            continue
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link", "")
        child = child_el.get("link", "")
        if not parent or not child:
            continue
        if child in parent_of:
            raise ValueError(
                f"Link '{child}' has multiple fixed-joint parents in "
                f"{urdf_path}: '{parent_of[child][0]}' and '{parent}'. "
                "URDF requires exactly one parent per link."
            )
        parent_of[child] = (parent, _origin_to_homogeneous(joint.find("origin")))
    return parent_of


def _compose_fixed_chain(
    parent_of: Dict[str, Tuple[str, np.ndarray]],
    link: str,
    base: str,
    urdf_path: Path,
) -> np.ndarray:
    """Walk the fixed-joint chain from ``link`` to ``base`` and compose ``T_base_to_link``.

    The result satisfies ``T_base_to_link = T_base_to_x_1 @ T_x_1_to_x_2 @ ... @ T_x_n_to_link``.
    Raises ``ValueError`` if the chain is broken, contains a non-fixed joint,
    or forms a cycle.
    """
    if link == base:
        return np.eye(4)
    T = np.eye(4)
    visited: Set[str] = set()
    cur = link
    while cur != base:
        visited.add(cur)
        if cur not in parent_of:
            raise ValueError(
                f"Link '{link}' has no fixed-joint chain to base '{base}' "
                f"in {urdf_path} (stuck at '{cur}')"
            )
        parent, T_parent_to_child = parent_of[cur]
        T = T_parent_to_child @ T
        if parent in visited:
            raise ValueError(
                f"Cycle detected in URDF chain from '{link}' to "
                f"'{base}' via '{parent}' in {urdf_path}"
            )
        cur = parent
    return T


def extract_tag_transforms(
    urdf_path: str | Path,
    prefix: str = "apriltag",
    base_link_name: Optional[str] = None,
    families: Optional[Iterable[str]] = None,
) -> Dict[Tuple[str, int], TagTransform]:
    """Walk the URDF joints and return ``base -> tag`` for every tag link.

    The returned ``T_base_to_tag`` is the **full** transform from the base
    link to the tag link, composing every parent->child fixed joint on
    the path. This means tags that are nested under an intermediate link
    (e.g. ``dock_point``) are handled transparently -- the joint's
    immediate parent does not need to be the base link.

    Parameters
    ----------
    urdf_path
        Path to the URDF XML file.
    prefix
        Link-name prefix used for tags (default ``"apriltag"``). The link
        name is expected to be ``f"{prefix}<family>_<id>"``.
    base_link_name
        Name of the object's base link. If ``None``, the URDF root link is
        used. Tags fixed-jointed to a sub-tree that does not descend from
        the base link are silently skipped.
    families
        Optional whitelist of family names (e.g. ``{"tag25h9", "tag36h11"}``).
        When ``None``, every family found in the URDF is returned.

    Returns
    -------
    dict
        ``{(family, tag_id): TagTransform}`` for every matching fixed joint.
        ``(family, tag_id)`` is unique because families encode both the
        family and the per-family numeric id.

    Raises
    ------
    FileNotFoundError
        If ``urdf_path`` does not exist.
    ValueError
        If no tag links are found, or if the URDF has no links at all.
    """
    urdf_path = Path(urdf_path).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    base = _resolve_base_link(urdf_path, base_link_name)
    link_names = _load_link_names(urdf_path)
    if base not in link_names:
        raise ValueError(f"Resolved base link '{base}' missing from URDF")

    whitelist: Optional[Set[str]] = None if families is None else {str(f) for f in families}

    # Build the fixed-joint parent map once; reuse it for every tag walk.
    parent_of = _build_fixed_joint_parent_map(urdf_path)

    result: Dict[Tuple[str, int], TagTransform] = {}
    for child in parent_of:
        parsed = _parse_tag_link(child, prefix)
        if parsed is None:
            continue
        family, tag_id = parsed
        if whitelist is not None and family not in whitelist:
            continue
        try:
            T = _compose_fixed_chain(parent_of, child, base, urdf_path)
        except ValueError:
            # Tag is not reachable from base via fixed joints; skip.
            continue
        key = (family, tag_id)
        if key in result:
            first = result[key].child_link
            raise ValueError(
                f"Duplicate tag id in {urdf_path}: both '{first}' and "
                f"'{child}' parse to (family={family}, id={tag_id}). "
                f"Tag ids are per-family and must be unique; rename one of "
                f"the joint/link pairs so the trailing <id> digits differ."
            )
        result[key] = TagTransform(
            tag_id=tag_id, family=family, child_link=child, T_base_to_tag=T
        )

    if not result:
        raise ValueError(
            f"No '{prefix}<family>_<id>' fixed-jointed links found in {urdf_path}"
        )
    return result


def link_transform_from_base(
    urdf_path: str | Path,
    link_name: str,
    base_link_name: Optional[str] = None,
) -> np.ndarray:
    """Return ``T_base_to_link`` for any link in the URDF.

    Walks the fixed-joint chain from ``link_name`` up to the base link,
    composing transforms along the way. Used for non-tag links (e.g. the
    dock point) whose pose is expressed in the tag object's base frame.

    Parameters
    ----------
    urdf_path
        Path to the URDF XML file.
    link_name
        Name of the link whose base-relative pose is wanted.
    base_link_name
        Name of the base link. If ``None``, the URDF root link is used.

    Returns
    -------
    np.ndarray
        4x4 homogeneous transform ``base -> link``.

    Raises
    ------
    FileNotFoundError
        If ``urdf_path`` does not exist.
    ValueError
        If ``link_name`` is not in the URDF, is not reachable from the
        base via fixed joints, or is reached via a non-``fixed`` joint.
    """
    urdf_path = Path(urdf_path).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    base = _resolve_base_link(urdf_path, base_link_name)
    link_names = _load_link_names(urdf_path)
    if link_name not in link_names:
        raise ValueError(
            f"Link '{link_name}' not in URDF {urdf_path}"
        )

    parent_of = _build_fixed_joint_parent_map(urdf_path)
    return _compose_fixed_chain(parent_of, link_name, base, urdf_path)