"""Approximate time sync of one camera's detector to another's (``sync_to``).

Why
---
Both cameras stream at 15 Hz, and each detector runs a 5 Hz timer that takes
the newest image. Two independent timers pick images up to a full detector
period (0.2 s) apart, so a ``cam_down`` detection is up to 0.2 s away from
the nearest ``cam_front`` keyframe it is attached to in dock_fuser (only the
EKF's motion over that interval bridges the gap).

With ``sync_to`` set, a detector runs no timer. Every message of the
reference detector (its ``apriltag_detection/tags``, stamped with the image
it processed) asks for *this* camera's image nearest that stamp, so the two
detections are at most half a camera frame apart (33 ms at 15 Hz), plus the
cameras' own clock offset. The keyframe rate stays the reference's, and
this detector processes exactly one image per reference message.

How a match is chosen (:class:`ImageSync`)
-----------------------------------------
Images and reference stamps arrive in any order. A reference is answered
with the buffered image of smallest ``|t_image - t_ref|`` once no better
one can come: an image newer than ``t_ref`` has arrived (later images are
only farther), or a reference ``wait`` seconds newer has arrived (this
camera is late or stalled).
A reference with no image within ``max_dt`` is dropped (counted), so a
stalled camera never produces a stale detection. An image is used at most
once.
"""

from __future__ import annotations

import bisect
from collections import deque
from typing import Any, Deque, List, Optional, Tuple


class ImageSync:
    """Match reference stamps to the nearest buffered image (see the module docstring).

    Parameters
    ----------
    max_dt
        Largest accepted ``|t_image - t_ref|`` [s]. Half a camera frame
        period plus some clock slack (15 Hz: 0.033 s, default 0.05 s).
    wait
        How long (in reference time: a newer reference this much later) to
        wait for a newer image before answering with the best older one [s].
    buffer_seconds
        Image history kept, in image time [s].
    """

    def __init__(self, max_dt: float = 0.05, wait: float = 0.1,
                 buffer_seconds: float = 2.0) -> None:
        self.max_dt = float(max_dt)
        self.wait = float(wait)
        self.buffer_seconds = float(buffer_seconds)
        self._stamps: List[float] = []
        self._images: List[Any] = []
        self._used: set = set()
        self._refs: Deque[float] = deque()
        self.n_unmatched = 0           # references with no image within max_dt

    def add_image(self, stamp: float, image: Any) -> None:
        """Buffer one image (any payload) by its stamp."""
        stamp = float(stamp)
        i = bisect.bisect_left(self._stamps, stamp)
        if i < len(self._stamps) and self._stamps[i] == stamp:
            return                     # re-delivered image
        self._stamps.insert(i, stamp)
        self._images.insert(i, image)
        cutoff = self._stamps[-1] - self.buffer_seconds
        n = bisect.bisect_left(self._stamps, cutoff)
        if n:
            dropped = set(self._stamps[:n])
            del self._stamps[:n]
            del self._images[:n]
            self._used -= dropped

    def add_reference(self, stamp: float) -> None:
        """Ask for this camera's image nearest ``stamp`` (a reference detection's image stamp)."""
        self._refs.append(float(stamp))

    @property
    def newest_image_stamp(self) -> Optional[float]:
        return self._stamps[-1] if self._stamps else None

    def pop_ready(self) -> List[Tuple[float, float, Any]]:
        """Answer every reference that can be decided now: ``[(t_ref, t_image, image)]``."""
        out: List[Tuple[float, float, Any]] = []
        newest = self.newest_image_stamp
        while self._refs:
            t_ref = self._refs[0]
            # Decided once an image newer than t_ref is in (later ones are only
            # farther), or the reference clock has moved `wait` past it (this
            # camera is late or stalled: answer with what is there, or drop).
            newer_image = newest is not None and newest > t_ref
            timed_out = self._refs[-1] - t_ref >= self.wait
            if not (newer_image or timed_out):
                break                  # a nearer, newer image may still come
            self._refs.popleft()
            best = self._nearest(t_ref)
            if best is None:
                self.n_unmatched += 1
                continue
            t_img = self._stamps[best]
            self._used.add(t_img)
            out.append((t_ref, t_img, self._images[best]))
        return out

    def flush(self) -> List[Tuple[float, float, Any]]:
        """Answer every waiting reference with what is buffered (end of a bag)."""
        out: List[Tuple[float, float, Any]] = []
        while self._refs:
            t_ref = self._refs.popleft()
            best = self._nearest(t_ref)
            if best is None:
                self.n_unmatched += 1
                continue
            self._used.add(self._stamps[best])
            out.append((t_ref, self._stamps[best], self._images[best]))
        return out

    def _nearest(self, t_ref: float) -> Optional[int]:
        i = bisect.bisect_left(self._stamps, t_ref)
        cands = [j for j in (i - 1, i) if 0 <= j < len(self._stamps)
                 and self._stamps[j] not in self._used]
        if not cands:
            return None
        j = min(cands, key=lambda j: abs(self._stamps[j] - t_ref))
        return j if abs(self._stamps[j] - t_ref) <= self.max_dt else None
