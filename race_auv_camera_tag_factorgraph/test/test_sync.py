"""ImageSync: the synced detector answers each reference with its nearest image."""

import pytest

from race_auv_camera_tag_factorgraph.sync import ImageSync

FRAME = 1.0 / 15.0          # both cameras at 15 Hz


def test_nearest_image_once_a_newer_one_arrived():
    s = ImageSync(max_dt=0.05)
    for i in range(5):                        # cam_down images at 0, 66, 133, 200, 266 ms
        s.add_image(i * FRAME, f"img{i}")
    s.add_reference(0.21)                     # cam_front processed its image at 210 ms
    (t_ref, t_img, img), = s.pop_ready()
    assert t_ref == 0.21 and img == "img3" and abs(t_img - 0.2) < 1e-9


def test_waits_while_this_camera_is_behind():
    """Reference newer than every buffered image: wait, a nearer image may still come."""
    s = ImageSync(max_dt=0.05)
    s.add_image(0.0, "a")
    s.add_image(FRAME, "b")                   # 66 ms
    s.add_reference(0.13)
    assert s.pop_ready() == []                # newest image (66 ms) is older than 130 ms
    s.add_image(2 * FRAME, "c")               # 133 ms arrives late
    (_, _, img), = s.pop_ready()
    assert img == "c"


def test_stalled_camera_times_out_on_reference_time():
    """No newer image: answered (or dropped) once a reference `wait` later arrives."""
    s = ImageSync(max_dt=0.05, wait=0.1)
    s.add_image(0.20, "last")
    s.add_reference(0.22)
    assert s.pop_ready() == []
    s.add_reference(0.42)                     # next cam_front detection, 0.2 s later
    assert [o[2] for o in s.pop_ready()] == ["last"]   # 0.22 answered with the 20 ms-old image
    assert s.n_unmatched == 0                 # 0.42 is the newest reference: still waiting
    s.add_reference(0.62)
    assert s.pop_ready() == [] and s.n_unmatched == 1  # 0.42: nothing within 50 ms -> dropped


def test_reference_without_close_image_is_dropped():
    s = ImageSync(max_dt=0.05)
    s.add_image(0.0, "a")
    s.add_image(0.3, "b")                     # a 300 ms hole (dropped frames)
    s.add_reference(0.15)
    assert s.pop_ready() == [] and s.n_unmatched == 1


def test_each_image_used_once_and_redelivery_ignored():
    s = ImageSync(max_dt=0.05)
    s.add_image(0.0, "a")
    s.add_image(0.0, "dup")                   # same stamp again: ignored
    s.add_image(FRAME, "b")
    s.add_reference(0.01)
    s.add_reference(0.02)                     # same nearest image ("a"): not reused
    out = s.pop_ready()
    assert [o[2] for o in out] == ["a", "b"]  # 0.02 gets "b", 46.7 ms away (inside max_dt)


def test_flush_answers_waiting_references():
    s = ImageSync(max_dt=0.05)
    s.add_image(0.10, "x")
    s.add_reference(0.12)
    assert s.pop_ready() == []
    assert [o[2] for o in s.flush()] == ["x"]


@pytest.mark.parametrize("offset", [0.0, 0.011, 0.023, 0.032])
def test_synced_offset_is_at_most_half_a_frame(offset):
    """Two 15 Hz cameras with any phase offset: matched images <= half a frame apart."""
    s = ImageSync(max_dt=0.05)
    for i in range(60):
        s.add_image(i * FRAME + offset, i)
    for k in range(1, 18):
        s.add_reference(k * 3 * FRAME)        # the reference detector: every 3rd frame, 5 Hz
    worst = max(abs(t_img - t_ref) for t_ref, t_img, _ in s.pop_ready())
    assert worst <= FRAME / 2 + 1e-9
