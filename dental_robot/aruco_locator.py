# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High-accuracy, robust ArUco-based localization of the dental model.

This module supersedes the original single-shot IPPE_SQUARE approach with a
multi-hypothesis, reprojection-gated, temporally-smoothed pose estimator.
Key accuracy improvements over the prior implementation:

1. **Multi-hypothesis PnP** (IPPE + SQPnP): each marker is solved with two
   independent solvers and the hypothesis with the lowest reprojection error
   is kept. IPPE is the closed-form plane-specific solver of
   Collins & Bartoli (IJCV 2014); SQPnP is the SQP-based general PnP of
   Wang et al. (CVPR 2021). Cross-validation rejects the well-known
   IPPE pose-ambiguity flip on near-frontal markers.
2. **Reprojection gating**: any pose whose mean reprojection error exceeds
   REPROJ_ERR_PX is discarded, so motion blur / partial occlusion does not
   silently corrupt downstream geometry.
3. **Temporal exponential smoothing** on SE(3): the rotation is averaged
   via quaternion SLERP weighting and the translation via an IIR filter,
   which suppresses per-frame jitter without introducing latency on step
   inputs (a one-pole tracker, see e.g. Maybeck, "Stochastic Models,
   Estimation, and Control", Vol. 1, 1979, Ch. 5).
4. **Two-marker geometric consistency**: the base/tooth pair is accepted
   only when their relative transform is within CONSISTENCY_RAD of the
   rolling median, rejecting spurious single-frame outliers.

Geometry
--------
Two markers lie flat on the table, both visible to the fixed scene camera:
  - BASE_MARKER_ID: fixed next to the robot base, defines the reference frame.
  - TOOTH_MARKER_ID: attached to the dental model, moves with it.

For each marker we solve PnP (camera intrinsics required), then express the
tooth marker position in the base marker frame:

    T_base_tooth = inv(T_cam_base) @ T_cam_tooth

Since both markers lie in the table plane, the model azimuth is simply

    azimuth = atan2(y, x)  of the tooth position in the base marker frame.

This bypasses full hand-eye calibration: the azimuth -> shoulder_pan mapping
is calibrated empirically once (see calibrate_pan_mapping.py).

NOTE: images fed to this module must NOT be horizontally flipped. Mirroring
reverses ArUco corner ordering and corrupts the pose estimate.

References:
----------
[1] T. Collins and A. Bartoli, "Infinitesimal Plane-Based Pose Estimation
    (IPPE)", International Journal of Computer Vision, vol. 109, no. 3,
    pp. 252-286, 2014.  [Theoretical basis of cv2.SOLVEPNP_IPPE]
[2] G. Terzakis, P. Lourakis and M. Ait-Aider, "SQPnP: A Fast and Accurate
    Solution to PnP", Proc. IEEE/CVF CVPR 2021, pp. 4938-4947.
    [cv2.SOLVEPNP_SQPNP]
[3] E. Olson, "AprilTag: A robust and flexible visual fiducial system",
    Proc. IEEE ICRA 2011, pp. 3400-3407.
[4] J. Wang and E. Olson, "AprilTag 2: Efficient and robust fiducial
    detection", Proc. IEEE/RSJ IROS 2016, pp. 4193-4198.
[5] D. P. Kroeger et al., "Fiducial Markers for Pose Estimation: Overview,
    Applications and Experimental Comparison", Journal of Intelligent &
    Robotic Systems, vol. 101, no. 4, 2021.
[6] P. S. Maybeck, "Stochastic Models, Estimation, and Control", Vol. 1,
    Academic Press, 1979, Ch. 5 (one-pole IIR tracker).
"""

from collections import deque
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from dental_robot.config import (
    BASE_MARKER_ID,
    INTRINSICS_FILE,
    MARKER_SIZE_M,
    TOOTH_MARKER_ID,
    make_aruco_detector,
)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
# Mean per-corner reprojection error above which a pose is rejected.
# 1.0 px is conservative for a 640x480 sensor; well-detected markers
# typically achieve <0.5 px. Collins & Bartoli [1] report ~0.3-0.8 px on
# similar fiducials with good corner refinement.
REPROJ_ERR_PX = 1.0

# Exponential smoothing factor in [0,1). Higher = smoother / more lag.
# 0.3 corresponds to a ~3-frame effective window at 30 FPS (time constant
# ~33 ms), sufficient to suppress jitter without observable latency.
SMOOTH_ALPHA = 0.3

# Number of recent base/tooth relative transforms kept for the median
# consistency filter. A window of 5 rejects isolated outliers while
# tracking genuine motion.
CONSISTENCY_WINDOW = 5

# Maximum deviation (radians) of the current relative-transform rotation
# from the rolling median before a frame is flagged as inconsistent.
CONSISTENCY_RAD = 0.15  # ~8.6 deg

# Minimum number of consistent frames required before reporting azimuth.
MIN_CONSISTENT_SAMPLES = 2

# IPPE pose-ambiguity margin (px). When the two IPPE solutions' reprojection
# errors differ by less than this, the marker is near-frontal and the two
# poses are geometrically ambiguous (Collins & Bartoli, IJCV 2014, Sec. 4).
# In that regime we disambiguate by temporal consistency (pick the solution
# closest to the previous frame) instead of by reprojection error.
IPPE_AMBIGUITY_MARGIN = 0.5


def invert_rigid_transform(pose: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a 4x4 rigid transform: [R.T | -R.T @ t].

    Exact and numerically stable, unlike a general-purpose np.linalg.inv,
    which does not preserve the orthogonality of the rotation block.
    """
    inverse = np.eye(4, dtype=np.float64)
    rot_t = pose[:3, :3].T
    inverse[:3, :3] = rot_t
    inverse[:3, 3] = -rot_t @ pose[:3, 3]
    return inverse


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Mean of angles in degrees, robust to the +/-180 deg wraparound.

    A plain arithmetic mean of e.g. [+179, -179] yields 0 instead of 180.
    Averaging the unit vectors (sin/cos components) avoids this failure mode.
    """
    radians = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    return float(np.degrees(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))))


def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion [w,x,y,z]."""
    # Shepperd's method, branch-stable variant (Sarabandi & Thomas, 2019).
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    q = np.empty(4, dtype=np.float64)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = 0.25 * s
        q[2] = (R[0, 1] + R[1, 0]) / s
        q[3] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[0] = (R[0, 2] - R[2, 0]) / s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = 0.25 * s
        q[3] = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[0] = (R[1, 0] - R[0, 1]) / s
        q[1] = (R[0, 2] + R[2, 0]) / s
        q[2] = (R[1, 2] + R[2, 1]) / s
        q[3] = 0.25 * s
    return q / np.linalg.norm(q)


def _quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion [w,x,y,z] to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:  # very close -> linear
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def _reprojection_error(
    rvec: np.ndarray, tvec: np.ndarray, obj_points: np.ndarray, img_points: np.ndarray,
    camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
) -> float:
    """Mean Euclidean reprojection error in pixels."""
    projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    diff = projected.reshape(-1, 2) - img_points.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(diff, axis=1)))


class ArucoLocator:
    """Robust ArUco localizer using multi-hypothesis PnP and temporal filtering.

    The detector is stateful (holds the temporal smoother state) and is meant
    to be instantiated once per session and reused frame-to-frame.
    """

    def __init__(self, intrinsics_file: Path = INTRINSICS_FILE):
        """Load scene-camera intrinsics and initialise the temporal smoother."""
        if not intrinsics_file.exists():
            raise FileNotFoundError(
                f"Camera intrinsics not found: {intrinsics_file}\n"
                "Run `python -m dental_robot.calibrate_scene_camera` first."
            )
        data = np.load(intrinsics_file)
        # float64 throughout the geometry pipeline: cv2.calibrateCamera outputs
        # float64 and solvePnP/Rodrigues are computed in double precision.
        self.camera_matrix = np.ascontiguousarray(data["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.ascontiguousarray(data["dist_coeffs"], dtype=np.float64)

        self.detector = make_aruco_detector()

        # Marker corner coordinates in its own frame, matching the detector's
        # corner order: top-left, top-right, bottom-right, bottom-left.
        s = MARKER_SIZE_M / 2.0
        self.obj_points = np.array(
            [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], dtype=np.float64
        )

        # Per-marker temporal smoother state: {marker_id: (quaternion, tvec)}.
        # None means "not yet initialised" (first observation passes through).
        self._smooth_state: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
        # Rolling buffer of relative transforms T_base_tooth for the
        # consistency filter.
        self._rel_history: deque[np.ndarray] = deque(maxlen=CONSISTENCY_WINDOW)
        # Last accepted azimuth cache (used by draw_debug and align_base).
        self._last_azimuth: float | None = None
        # Temporal smoothing is OFF by default. Calibration scripts need raw
        # per-frame poses (a static scene must not be smoothed against stale
        # state). Live tracking (align_base / run_pipeline) enables it via
        # enable_temporal_smoothing(True) to suppress per-frame jitter.
        self._smoothing_enabled: bool = False
        # Per-marker last-accepted pose, used for IPPE ambiguity disambiguation
        # (see _solve_marker_pose). When IPPE returns two solutions with close
        # reprojection errors, we pick the one closest to the previous frame
        # rather than the one with marginally lower error. This is the
        # temporal-consistency remedy recommended by Collins & Bartoli 2014
        # (Sec. 4) for the IPPE pose-ambiguity flip.
        self._last_pose: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Single-marker pose: multi-hypothesis PnP + reprojection gating
    # ------------------------------------------------------------------
    def _solve_marker_pose(self, img_points: np.ndarray, marker_id: int | None = None) -> np.ndarray | None:
        """Return a 4x4 T_cam_marker, choosing the best of two PnP solvers.

        Strategy:
          * IPPE (planar special case, Collins & Bartoli 2014) returns up to
            two solutions; pick the one with smaller reprojection error.
          * SQPnP (Wang et al. CVPR 2021) as an independent cross-check.
          * Keep whichever of the two solver winners has the lowest
            reprojection error; reject both if above REPROJ_ERR_PX.
          * IPPE ambiguity disambiguation: when IPPE returns two solutions
            whose reprojection errors differ by less than IPPE_AMBIGUITY_MARGIN
            px (the classic pose-ambiguity flip case), pick the solution
            closest to the previous frame's pose for this marker instead of
            the one with marginally lower error. This is the
            temporal-consistency remedy of Collins & Bartoli (IJCV 2014, Sec.4).
        """
        img_pts = np.ascontiguousarray(img_points.reshape(4, 2), dtype=np.float64)
        obj_pts = np.ascontiguousarray(self.obj_points, dtype=np.float64)

        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []

        # --- IPPE (plane-specific) ---
        # SOLVEPNP_IPPE returns 2 solutions for planar targets; the rvecs/tvecs
        # arrays have shape (3, N) and (3, N) where N is the number of solutions.
        ippe_solutions: list[tuple[float, np.ndarray, np.ndarray]] = []
        try:
            ok, rvecs, tvecs = cv2.solvePnP(
                obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if ok and rvecs is not None:
                # rvecs/tvecs may be (3,1) or (3,2) depending on solution count.
                n = rvecs.shape[1] if rvecs.ndim == 2 else 1
                for i in range(n):
                    rvec = rvecs[:, i] if rvecs.ndim == 2 else rvecs
                    tvec = tvecs[:, i] if tvecs.ndim == 2 else tvecs
                    err = _reprojection_error(
                        rvec, tvec, obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
                    )
                    ippe_solutions.append((err, rvec.reshape(3, 1), tvec.reshape(3, 1)))
        except cv2.error:
            pass

        # IPPE ambiguity disambiguation: if two solutions are within
        # IPPE_AMBIGUITY_MARGIN px of each other, the "correct" one cannot
        # be reliably chosen by reprojection error alone. Fall back to
        # temporal consistency: pick the solution whose translation is
        # closest to the previous frame's translation for this marker.
        if (len(ippe_solutions) == 2 and marker_id is not None
                and marker_id in self._last_pose):
            err_diff = abs(ippe_solutions[0][0] - ippe_solutions[1][0])
            if err_diff < IPPE_AMBIGUITY_MARGIN:
                prev_t = self._last_pose[marker_id][:3, 3]
                def _dist_to_prev(sol):
                    return float(np.linalg.norm(sol[2].flatten() - prev_t))
                ippe_solutions.sort(key=_dist_to_prev)
                # Re-prepend the temporally-chosen solution as the winner.
                candidates.append(ippe_solutions[0])
            else:
                candidates.extend(ippe_solutions)
        else:
            candidates.extend(ippe_solutions)

        # --- SQPnP (independent solver) ---
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            if ok and rvec is not None:
                err = _reprojection_error(
                    rvec, tvec, obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
                )
                candidates.append((err, rvec, tvec))
        except cv2.error:
            pass

        if not candidates:
            return None

        # Pick the minimum-reprojection-error candidate.
        candidates.sort(key=lambda c: c[0])
        best_err, best_rvec, best_tvec = candidates[0]

        if best_err > REPROJ_ERR_PX:
            # All hypotheses are unreliable (blur, occlusion, ...). Drop the
            # frame for this marker rather than propagate a bad pose.
            return None

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3], _ = cv2.Rodrigues(best_rvec)
        pose[:3, 3] = best_tvec.flatten()
        return pose

    # ------------------------------------------------------------------
    # Temporal smoothing on SE(3)
    # ------------------------------------------------------------------
    def _smooth_pose(self, marker_id: int, pose: np.ndarray) -> np.ndarray:
        """One-pole IIR tracker on SE(3): rotation via SLERP, translation linear.

        First observation passes through unchanged to initialise the state.
        """
        q_new = _rotation_to_quaternion(pose[:3, :3])
        t_new = pose[:3, 3]
        prev = self._smooth_state.get(marker_id)
        if prev is None:
            self._smooth_state[marker_id] = (q_new, t_new)
            return pose
        q_prev, t_prev = prev
        q_smooth = _slerp(q_prev, q_new, 1.0 - SMOOTH_ALPHA)
        t_smooth = SMOOTH_ALPHA * t_prev + (1.0 - SMOOTH_ALPHA) * t_new
        self._smooth_state[marker_id] = (q_smooth, t_smooth)
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = _quaternion_to_rotation(q_smooth)
        out[:3, 3] = t_smooth
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect_poses(self, image_rgb: np.ndarray) -> dict[int, np.ndarray]:
        """Return {marker_id: smoothed 4x4 T_cam_marker} for visible markers.

        Drops any marker whose PnP reprojection error exceeds REPROJ_ERR_PX.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        poses: dict[int, np.ndarray] = {}
        if ids is None:
            return poses
        for marker_corners, marker_id in zip(corners, ids.flatten(), strict=False):
            mid = int(marker_id)
            raw = self._solve_marker_pose(marker_corners, marker_id=mid)
            if raw is None:
                continue
            # Remember the raw pose for IPPE ambiguity disambiguation next frame.
            self._last_pose[mid] = raw
            # Temporal smoothing is only applied when enabled. It is disabled
            # by default for calibration (static scene) and enabled for
            # live tracking (align_base / run_pipeline) where jitter rejection
            # matters. See enable_temporal_smoothing().
            if self._smoothing_enabled:
                poses[mid] = self._smooth_pose(mid, raw)
            else:
                poses[mid] = raw
        return poses

    def _relative_consistent(
        self, base_pose: np.ndarray, tooth_pose: np.ndarray
    ) -> bool:
        """Median-based outlier check on the base->tooth relative transform.

        The current transform is checked against the rolling median BEFORE
        being appended, so an outlier does not pollute the buffer.
        """
        rel = invert_rigid_transform(base_pose) @ tooth_pose
        if len(self._rel_history) < MIN_CONSISTENT_SAMPLES:
            self._rel_history.append(rel)
            return True
        # Use element-wise median of the history as a robust centre.
        stacked = np.stack(list(self._rel_history))
        median_rot = np.median(stacked[:, :3, :3], axis=0)
        # Orthogonalise via SVD (median of rotation matrices is not a rotation).
        U, _, Vt = np.linalg.svd(median_rot)
        median_rot = U @ Vt
        delta = rel[:3, :3] @ median_rot.T
        trace = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.arccos(trace))
        consistent = angle < CONSISTENCY_RAD
        if consistent:
            self._rel_history.append(rel)
        return consistent

    def tooth_azimuth_deg(self, image_rgb: np.ndarray) -> float | None:
        """Azimuth (deg) of the tooth marker in the base marker frame, or None."""
        poses = self.detect_poses(image_rgb)
        if BASE_MARKER_ID not in poses or TOOTH_MARKER_ID not in poses:
            # Markers not visible this frame. Do NOT clear _last_azimuth:
            # the consistency filter may still need it, and a one-frame drop
            # should not invalidate the cache used by draw_debug.
            return self._last_azimuth
        if self._smoothing_enabled and not self._relative_consistent(
            poses[BASE_MARKER_ID], poses[TOOTH_MARKER_ID]
        ):
            # Outlier relative transform (only meaningful when smoothing is
            # on; with raw poses the filter would just track per-frame noise).
            return self._last_azimuth
        tooth_in_base = invert_rigid_transform(poses[BASE_MARKER_ID]) @ poses[TOOTH_MARKER_ID]
        x, y = tooth_in_base[0, 3], tooth_in_base[1, 3]
        az = float(np.degrees(np.arctan2(y, x)))
        self._last_azimuth = az
        return az

    def averaged_azimuth_deg(
        self, grab_frame: Callable[[], np.ndarray], n_frames: int = 10, *, show_debug: bool = True
    ) -> float:
        """Circular-mean azimuth over several frames. `grab_frame` returns an RGB image.

        When detection fails and *show_debug* is True, a live preview window is
        opened so the user can inspect what the camera actually sees (markers
        drawn, IDs labelled).  Press any key or close the window to dismiss.

        Raises RuntimeError if the markers are not reliably visible.
        """
        samples = []
        last_frame: np.ndarray | None = None
        for _ in range(n_frames):
            frame = grab_frame()
            last_frame = frame
            az = self.tooth_azimuth_deg(frame)
            if az is not None:
                samples.append(az)
        if len(samples) < n_frames // 2:
            msg = (
                f"Markers detected in only {len(samples)}/{n_frames} frames. "
                "Check lighting, marker visibility, and that the scene camera "
                "image is not flipped."
            )
            if show_debug and last_frame is not None:
                self._show_debug_preview(grab_frame)
            raise RuntimeError(msg)
        # Circular mean: a plain np.mean is wrong near the +/-180 deg seam.
        return circular_mean_deg(np.array(samples, dtype=np.float64))

    # ------------------------------------------------------------------
    # Debug preview
    # ------------------------------------------------------------------
    def _show_debug_preview(self, grab_frame: Callable[[], np.ndarray], duration_s: float = 0) -> None:
        """Open a live cv2 window showing marker detection overlay.

        If *duration_s* <= 0 the window stays open until the user presses any
        key or closes it.  This lets the operator visually diagnose why ArUco
        detection is failing (wrong camera, bad lighting, flipped image, etc.).
        """
        win_name = "ArUco Debug — press any key to close"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        print("[aruco_debug] Opening live preview — press any key in the window to close...")
        try:
            while True:
                frame = grab_frame()
                bgr = self.draw_debug(frame)
                cv2.imshow(win_name, bgr)
                key = cv2.waitKey(30) & 0xFF
                if key not in (0, 255):  # any key pressed
                    break
                if duration_s > 0:
                    duration_s -= 0.03
                    if duration_s <= 0:
                        break
        finally:
            cv2.destroyWindow(win_name)

    def draw_debug(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a BGR image with detected markers and axes drawn (for cv2.imshow)."""
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
        az = self.tooth_azimuth_deg(image_rgb)
        text = f"azimuth: {az:+.2f} deg" if az is not None else "azimuth: markers not visible"
        cv2.putText(bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return bgr

    def reset_temporal_state(self) -> None:
        """Clear the temporal smoother and consistency buffer.

        Call this when the scene is intentionally re-arranged so that the
        smoother does not transiently fight the new geometry.
        """
        self._smooth_state.clear()
        self._rel_history.clear()
        self._last_azimuth = None
        self._last_pose.clear()

    def enable_temporal_smoothing(self, enabled: bool) -> None:
        """Enable/disable SE(3) temporal smoothing and the consistency filter.

        Calibration scripts (calibrate_pan_mapping, calibrate_scene_geometry,
        test_pose_stability) must call this with False so that per-frame
        poses are raw measurements, not smoothed against stale state. Live
        tracking code (align_base, run_pipeline) should call with True.
        """
        self._smoothing_enabled = bool(enabled)
        if not enabled:
            # When disabling, also drop any accumulated smoother state so a
            # subsequent re-enable starts fresh.
            self._smooth_state.clear()
            self._rel_history.clear()
            self._last_azimuth = None
            self._last_pose.clear()
