import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

# -------------------------
# utils
# -------------------------
def rodrigues_to_Rt(rvec, tvec) -> np.ndarray:
    """OpenCV rvec,tvec (Obj->Cam) => 4x4 T_C_O"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T

def inv_T(T: np.ndarray) -> np.ndarray:
    """Inverse of 4x4 rigid transform"""
    R = T[:3, :3]
    t = T[:3, 3:4]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3:4] = -R.T @ t
    return Ti

def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula from axis-angle to R"""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

# -------------------------
# Cube config
# -------------------------
@dataclass
class CubeConfig:
    cube_side_m: float = 0.03       # cube size
    marker_size_m: float = 0.022     # marker size
    dictionary_name: str = "DICT_4X4_50"
    marker_ids: tuple = (0, 1, 2, 3, 4)

    id_to_face: dict = None
    face_roll_deg: dict = None

    def __post_init__(self):
        if self.id_to_face is None:
            self.id_to_face = {
                0: "+Z",
                1: "+X",
                2: "+Y",
                3: "-X",
                4: "-Y",
            }
        if self.face_roll_deg is None:
            # Fixed to the validated real attachment setup.
            self.face_roll_deg = {
                0: 0.0,
                1: 270.0,
                2: 0.0,
                3: 90.0,
                4: 180.0,
            }

# -------------------------
# Cube geometry
# -------------------------
class ArucoCubeModel:
    def __init__(self, cfg: CubeConfig):
        self.cfg = cfg
        d = cfg.cube_side_m / 2.0
        s = cfg.marker_size_m / 2.0

        # Each face: center c, axes u,v, normal n (in rig/object coord)
        self.face_defs = {
            "+Z": (np.array([0, 0,  d], np.float64),
                   np.array([1, 0, 0], np.float64),   # u = +X
                   np.array([0, 1, 0], np.float64),   # v = +Y
                   np.array([0, 0, 1], np.float64)),  # n = +Z

            "-Z": (np.array([0, 0, -d], np.float64),
                   np.array([1, 0, 0], np.float64),   # u = +X
                   np.array([0, 1, 0], np.float64),   # v = +Y
                   np.array([0, 0,-1], np.float64)),  # n = -Z

            "+X": (np.array([ d, 0, 0], np.float64),
                   np.array([0, 0,-1], np.float64),   # u = -Z
                   np.array([0, 1, 0], np.float64),   # v = +Y
                   np.array([1, 0, 0], np.float64)),  # n = +X

            "-X": (np.array([-d, 0, 0], np.float64),
                   np.array([0, 0, 1], np.float64),   # u = +Z
                   np.array([0, 1, 0], np.float64),   # v = +Y
                   np.array([-1,0, 0], np.float64)),  # n = -X

            "+Y": (np.array([0, d, 0], np.float64),
                   np.array([1, 0, 0], np.float64),   # u = +X
                   np.array([0, 0,-1], np.float64),   # v = -Z
                   np.array([0, 1, 0], np.float64)),  # n = +Y

            "-Y": (np.array([0,-d, 0], np.float64),
                   np.array([1, 0, 0], np.float64),   # u = +X
                   np.array([0, 0, 1], np.float64),   # v = +Z
                   np.array([0,-1, 0], np.float64)),  # n = -Y
        }

        # Marker local corners on marker plane (z=0 in local marker frame).
        # Fixed corner index order used by calibration:
        #   0=(+x,-y), 1=(-x,-y), 2=(-x,+y), 3=(+x,+y)
        self.local_corners = np.array([
            [ s, -s, 0],
            [-s, -s, 0],
            [-s,  s, 0],
            [ s,  s, 0],
        ], dtype=np.float64)
    def marker_corners_in_rig(self, marker_id: int) -> np.ndarray:
        marker_id = int(marker_id)
        face = self.cfg.id_to_face[marker_id]
        c, u, v, n = self.face_defs[face]

        roll_deg = float(self.cfg.face_roll_deg.get(marker_id, 0.0))
        roll = np.deg2rad(roll_deg)
        Rr = rot_axis_angle(n, roll)

        u2 = (Rr @ u.reshape(3, 1)).reshape(3)
        v2 = (Rr @ v.reshape(3, 1)).reshape(3)

        pts = []
        for p in self.local_corners:
            pts.append(c + u2 * p[0] + v2 * p[1])
        return np.asarray(pts, dtype=np.float64)

# -------------------------
# Target
# -------------------------
class ArucoCubeTarget:
    def __init__(self, cfg: CubeConfig):
        self.cfg = cfg
        self.model = ArucoCubeModel(cfg)

        d = getattr(cv2.aruco, cfg.dictionary_name)
        self.dictionary = cv2.aruco.getPredefinedDictionary(d)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

    def detect(self, bgr) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Return (corners_list, ids_flat or None)"""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return [], None
        return corners, ids.flatten().astype(int)

    # 공개 wrapper (Step3 overlay, 디버그에서 쓰기 위함)
    def build_correspondences(
        self,
        corners_list,
        ids,
        min_markers: int,
        only_ids: Optional[List[int]] = None
    ):
        return self._build_correspondences(corners_list, ids, min_markers, only_ids=only_ids)

    def _build_correspondences(self, corners_list, ids, min_markers, only_ids=None):
        obj_pts, img_pts, used = [], [], []
        only_ids = set(only_ids) if only_ids is not None else None

        for c, mid in zip(corners_list, ids):
            mid = int(mid)
            if mid not in self.cfg.id_to_face:
                continue
            if only_ids is not None and mid not in only_ids:
                continue

            obj = self.model.marker_corners_in_rig(mid)  # (4,3)
            img = c.reshape(4, 2)                        # (4,2)

            obj_pts.append(obj)
            img_pts.append(img)
            used.append(mid)

        if len(used) < int(min_markers):
            return None, None, used

        obj_pts = np.concatenate(obj_pts).reshape(-1, 1, 3).astype(np.float64)
        img_pts = np.concatenate(img_pts).reshape(-1, 1, 2).astype(np.float64)
        return obj_pts, img_pts, used

    def _solve_and_score(self, obj_pts, img_pts, K, D, use_ransac: bool, marker_id: Optional[int] = None):
        n = int(obj_pts.shape[0])

        if n >= 8:
            # 마커 2개 이상: ITERATIVE (모호성 없음)
            flags = cv2.SOLVEPNP_ITERATIVE
            if use_ransac:
                ok, rvec, tvec, _ = cv2.solvePnPRansac(
                    obj_pts, img_pts, K, D,
                    flags=flags,
                    reprojectionError=5.0,
                    iterationsCount=200,
                    confidence=0.999
                )
            else:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=flags)
            if not ok:
                return None
        else:
            # 마커 1개(4점): IPPE가 복수 해를 반환하므로 물리 조건으로 해 선택
            # - z>0 (큐브 원점이 카메라 앞)
            # - (가능한 경우) 관측 face의 outward normal이 카메라를 향함
            retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE
            )
            if retval == 0 or len(rvecs) == 0:
                return None

            candidates = []
            for i in range(len(rvecs)):
                rv = np.asarray(rvecs[i], dtype=np.float64).reshape(3, 1)
                tv = np.asarray(tvecs[i], dtype=np.float64).reshape(3, 1)
                proj2, _ = cv2.projectPoints(obj_pts.reshape(-1, 3), rv, tv, K, D)
                proj2 = proj2.reshape(-1, 2)
                err = np.linalg.norm(
                    proj2 - img_pts.reshape(-1, 2), axis=1
                ).astype(np.float64)
                err_mean = float(np.mean(err)) if err.size else float("inf")

                z_ok = float(tv[2]) > 0.0
                vis_ok = False
                vis_score = -1e12

                # marker_id를 알면 face normal 가시성으로 해 모호성 추가 제거
                if marker_id is not None and marker_id in self.cfg.id_to_face:
                    face = self.cfg.id_to_face[int(marker_id)]
                    c_face, _, _, n_face = self.model.face_defs[face]
                    R, _ = cv2.Rodrigues(rv)
                    n_cam = (R @ n_face.reshape(3, 1)).reshape(3)
                    c_cam = (R @ c_face.reshape(3, 1)).reshape(3) + tv.reshape(3)
                    # face center -> camera(origin) 벡터와 법선의 내적
                    # 관측 가능한 face라면 양수여야 함
                    vis_score = float(np.dot(n_cam, -c_cam))
                    vis_ok = vis_score > 0.0

                candidates.append(
                    dict(
                        rvec=rv,
                        tvec=tv,
                        proj2=proj2,
                        err=err,
                        err_mean=err_mean,
                        z_ok=z_ok,
                        vis_ok=vis_ok,
                        vis_score=vis_score,
                    )
                )

            if marker_id is not None and marker_id in self.cfg.id_to_face:
                # 1) z_ok & vis_ok  2) z_ok  3) fallback
                def rank(c):
                    tier = 2
                    if c["z_ok"] and c["vis_ok"]:
                        tier = 0
                    elif c["z_ok"]:
                        tier = 1
                    return (tier, c["err_mean"], -c["vis_score"])
            else:
                # marker_id 정보 없으면 기존 제약: z_ok 우선 + reproj 최소
                def rank(c):
                    tier = 0 if c["z_ok"] else 1
                    return (tier, c["err_mean"])

            best = min(candidates, key=rank)
            return dict(
                rvec=best["rvec"],
                tvec=best["tvec"],
                proj2=best["proj2"],
                err=best["err"],
            )

        proj2, _ = cv2.projectPoints(obj_pts.reshape(-1, 3), rvec, tvec, K, D)
        proj2 = proj2.reshape(-1, 2)
        err = np.linalg.norm(proj2 - img_pts.reshape(-1, 2), axis=1).astype(np.float64)

        return dict(rvec=rvec, tvec=tvec, proj2=proj2, err=err)

    def solve_pnp_cube(
        self,
        bgr,
        K,
        D,
        use_ransac: bool = True,
        min_markers: int = 1,
        reproj_thr_mean_px: float = 10.0,
        only_ids: Optional[List[int]] = None,
        mean_err_max_px: Optional[float] = None,
        single_marker_only: bool = False,
        return_reproj: bool = False,
    ):
        """
        기본 동작:
          - 다중 마커(2개 이상)로 PnP 시도
          - 실패/품질 미달이면 단일 마커(best-of-single) fallback

        single_marker_only=True:
          - 프레임 내 감지된 마커 중 reprojection mean이 가장 작은
            단일 마커 1개만 사용

        Returns:
          - if return_reproj False:
              (ok, rvec, tvec, used)
          - if return_reproj True:
              (ok, rvec, tvec, used, reproj_dict)
            reproj_dict contains: obj_pts,img_pts,proj2,err,err_mean,err_median,err_p90,n_points,rvec,tvec
        """
        corners_list, ids = self.detect(bgr)
        if ids is None:
            return (False, None, None, [], None) if return_reproj else (False, None, None, [])

        only_ids_set = set(only_ids) if only_ids is not None else None

        # 1) 다중 마커(2개 이상) 우선 시도 (single_marker_only=False일 때만)
        multi_sol = None
        multi_used = []
        obj_all, img_all, used_all = None, None, None
        if not bool(single_marker_only):
            obj_all, img_all, used_all = self._build_correspondences(
                corners_list, ids, min_markers=max(2, int(min_markers)), only_ids=only_ids
            )
            if obj_all is not None and used_all is not None and len(used_all) >= 2:
                sol = self._solve_and_score(obj_all, img_all, K, D, use_ransac, marker_id=None)
                if sol is not None:
                    multi_sol = sol
                    multi_used = used_all

        # 2) 단일 마커 fallback (best-of-single)
        best_sol     = None
        best_err     = float("inf")
        best_used    = []
        best_obj_pts = None
        best_img_pts = None

        for c, mid in zip(corners_list, ids):
            mid = int(mid)
            if mid not in self.cfg.id_to_face:
                continue
            if only_ids_set is not None and mid not in only_ids_set:
                continue

            obj_pts, img_pts, used = self._build_correspondences([c], [mid], 1)
            if obj_pts is None:
                continue

            sol = self._solve_and_score(obj_pts, img_pts, K, D, use_ransac, marker_id=mid)
            if sol is None:
                continue

            err_mean = float(np.mean(sol["err"]))
            if err_mean < best_err:
                best_err     = err_mean
                best_sol     = sol
                best_used    = used
                best_obj_pts = obj_pts
                best_img_pts = img_pts

        # 최종 해 선택
        if bool(single_marker_only):
            chosen_sol = best_sol
            chosen_used = best_used
            chosen_obj_pts = best_obj_pts
            chosen_img_pts = best_img_pts
        else:
            # - 다중 마커 해가 있고 reproj 임계 이내면 다중 해 채택
            # - 아니면 단일 해 채택
            if multi_sol is not None:
                err_multi = multi_sol["err"]
                err_multi_mean = float(np.mean(err_multi)) if err_multi.size else float("inf")
                if err_multi_mean <= float(reproj_thr_mean_px):
                    chosen_sol = multi_sol
                    chosen_used = list(multi_used)
                    chosen_obj_pts = obj_all
                    chosen_img_pts = img_all
                else:
                    chosen_sol = best_sol
                    chosen_used = best_used
                    chosen_obj_pts = best_obj_pts
                    chosen_img_pts = best_img_pts
            else:
                chosen_sol = best_sol
                chosen_used = best_used
                chosen_obj_pts = best_obj_pts
                chosen_img_pts = best_img_pts

        if chosen_sol is None:
            return (False, None, None, [], None) if return_reproj else (False, None, None, [])

        err = chosen_sol["err"]
        reproj = {
            "obj_pts":    chosen_obj_pts,
            "img_pts":    chosen_img_pts,
            "proj2":      chosen_sol["proj2"],
            "err":        err,
            "err_mean":   float(np.mean(err)) if err.size else float("inf"),
            "err_median": float(np.median(err)) if err.size else float("inf"),
            "err_p90":    float(np.percentile(err, 90)) if err.size else float("inf"),
            "n_points":   int(err.size),
            "rvec":       chosen_sol["rvec"],
            "tvec":       chosen_sol["tvec"],
        }

        max_thr  = float("inf") if mean_err_max_px is None else float(mean_err_max_px)
        ok_final = (reproj["err_mean"] <= float(reproj_thr_mean_px)) and (reproj["err_mean"] <= max_thr)

        if return_reproj:
            return ok_final, chosen_sol["rvec"], chosen_sol["tvec"], chosen_used, reproj
        return ok_final, chosen_sol["rvec"], chosen_sol["tvec"], chosen_used

    # ------------------------------------------------------------------ #
    # Depth support
    # ------------------------------------------------------------------ #

    def sample_depth_at_corners(
        self,
        depth_u16: np.ndarray,
        depth_scale: float,
        corners_list: List[np.ndarray],
        ids: np.ndarray,
        radius: int = 2,
    ) -> Dict[int, np.ndarray]:
        """
        감지된 마커 코너 위치에서 depth 값(m)을 샘플링한다.

        Args:
            depth_u16:   uint16 depth 이미지 (H, W) - align_depth_to_color 적용 후
            depth_scale: depth scale (m/unit), intrinsics npz의 depth_scale_m_per_unit
            corners_list, ids: cube.detect() 반환값
            radius:      코너 주변 패치 반경(px), median 사용

        Returns:
            {marker_id: (4,) float64 depth_m array,  NaN = 유효 depth 없음}
        """
        h, w = depth_u16.shape[:2]
        result: Dict[int, np.ndarray] = {}

        for c, mid in zip(corners_list, ids):
            mid = int(mid)
            if mid not in self.cfg.id_to_face:
                continue
            corners_px = c.reshape(4, 2)
            depths = []
            for (u, v) in corners_px:
                r0 = max(0, int(v) - radius)
                r1 = min(h, int(v) + radius + 1)
                c0 = max(0, int(u) - radius)
                c1 = min(w, int(u) + radius + 1)
                patch = depth_u16[r0:r1, c0:c1].astype(np.float64)
                valid = patch[patch > 0]
                if valid.size:
                    depths.append(float(np.median(valid)) * float(depth_scale))
                else:
                    depths.append(float("nan"))
            result[mid] = np.array(depths, dtype=np.float64)

        return result

    def get_3d_correspondences_from_depth(
        self,
        depth_u16: np.ndarray,
        depth_scale: float,
        corners_list: List[np.ndarray],
        ids: np.ndarray,
        K: np.ndarray,
        radius: int = 2,
        min_valid_per_marker: int = 3,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[int]]:
        """
        depth backprojection으로 마커 코너의 3D 좌표(카메라 프레임)를 계산하고,
        rig 프레임 기준 3D 좌표와 대응점 쌍을 반환한다.

        Returns:
            obj_pts:  (N, 3) rig/object 프레임 3D 점
            cam_pts:  (N, 3) 카메라 프레임 3D 점 (depth backprojection)
            used_ids: 유효 depth 코너가 min_valid_per_marker 이상인 marker id 목록
        """
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        h, w = depth_u16.shape[:2]
        obj_list: List[np.ndarray] = []
        cam_list: List[List[float]] = []
        used_ids: List[int] = []

        for c, mid in zip(corners_list, ids):
            mid = int(mid)
            if mid not in self.cfg.id_to_face:
                continue
            obj_corners = self.model.marker_corners_in_rig(mid)  # (4, 3)
            img_corners = c.reshape(4, 2)

            valid_obj: List[np.ndarray] = []
            valid_cam: List[List[float]] = []

            for i, (u, v) in enumerate(img_corners):
                r0 = max(0, int(v) - radius)
                r1 = min(h, int(v) + radius + 1)
                c0 = max(0, int(u) - radius)
                c1 = min(w, int(u) + radius + 1)
                patch = depth_u16[r0:r1, c0:c1].astype(np.float64)
                valid = patch[patch > 0]
                if not valid.size:
                    continue
                z = float(np.median(valid)) * float(depth_scale)
                x = (float(u) - cx) * z / fx
                y = (float(v) - cy) * z / fy
                valid_obj.append(obj_corners[i])
                valid_cam.append([x, y, z])

            if len(valid_cam) >= int(min_valid_per_marker):
                obj_list.extend(valid_obj)
                cam_list.extend(valid_cam)
                used_ids.append(mid)

        if len(cam_list) < 3:
            return None, None, []

        return (
            np.array(obj_list, dtype=np.float64),
            np.array(cam_list, dtype=np.float64),
            used_ids,
        )

    @staticmethod
    def _kabsch_svd(
        obj_pts: np.ndarray,
        cam_pts: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Kabsch / Umeyama SVD로 3D-3D 대응점에서 rigid transform T_C_O (4x4) 계산.

        cam_pts ≈ R @ obj_pts + t  (column-vector convention)

        Args:
            obj_pts: (N, 3) rig 프레임 점
            cam_pts: (N, 3) 카메라 프레임 점
        Returns:
            T_C_O (4x4 float64) or None (수치 오류)
        """
        if len(obj_pts) < 3:
            return None
        mu_o = obj_pts.mean(axis=0)
        mu_c = cam_pts.mean(axis=0)
        A = (obj_pts - mu_o).T @ (cam_pts - mu_c)  # (3, 3)
        U, _, Vt = np.linalg.svd(A)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, float(d)])
        R = Vt.T @ D @ U.T
        t = mu_c - R @ mu_o
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def solve_pose_from_depth(
        self,
        bgr: np.ndarray,
        depth_u16: np.ndarray,
        depth_scale: float,
        K: np.ndarray,
        D: np.ndarray,
        min_valid_pts: int = 6,
        radius: int = 2,
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], List[int], str]:
        """
        depth backprojection + Kabsch SVD로 큐브 pose 추정.
        유효 depth 점 부족 시 일반 PnP fallback.

        Returns:
            (ok, rvec, tvec, used_ids, method)
            method: "depth_svd" | "pnp_fallback" | "no_markers"
        """
        corners_list, ids = self.detect(bgr)
        if ids is None:
            return False, None, None, [], "no_markers"

        obj_pts, cam_pts, used_ids = self.get_3d_correspondences_from_depth(
            depth_u16, depth_scale, corners_list, ids, K, radius=radius
        )

        if obj_pts is not None and len(obj_pts) >= int(min_valid_pts):
            T = self._kabsch_svd(obj_pts, cam_pts)
            if T is not None and float(T[2, 3]) > 0.0:
                R = T[:3, :3]
                t = T[:3, 3]
                rvec, _ = cv2.Rodrigues(R)
                return True, rvec, t.reshape(3, 1), used_ids, "depth_svd"

        # fallback
        ok, rvec, tvec, used = self.solve_pnp_cube(bgr, K, D, min_markers=1)
        return ok, rvec, tvec, used, "pnp_fallback"

    def project_all_markers(self, rvec, tvec, K, D) -> Dict[int, np.ndarray]:
        """
        PnP로 구한 큐브 pose(rvec, tvec)를 이용해
        모든 마커의 코너를 이미지에 투영한다.
        Returns: {marker_id: (4, 2) projected corners}
        """
        result = {}
        for mid in self.cfg.id_to_face:
            obj = self.model.marker_corners_in_rig(mid).reshape(-1, 1, 3)
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
            result[mid] = proj.reshape(4, 2)
        return result
