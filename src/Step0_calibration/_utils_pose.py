# utils_pose.py
import numpy as np

def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """ (w,x,y,z), numerically stable """
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)

def _quat_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R

def _rot_angle(R: np.ndarray) -> float:
    tr = np.trace(R)
    c = (tr - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.arccos(c))

def _average_quats(quats: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """ Markley method: eig of weighted outer products """
    A = np.zeros((4, 4), dtype=np.float64)
    for q, w in zip(quats, weights):
        q = q.reshape(4, 1)
        A += w * (q @ q.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    # canonical sign
    if q_avg[0] < 0:
        q_avg = -q_avg
    return q_avg / (np.linalg.norm(q_avg) + 1e-12)

def se3_distance(Ta: np.ndarray, Tb: np.ndarray, w_rot=1.0, w_trans=1.0) -> float:
    """ simple metric: rot_angle + trans_norm """
    Ra, ta = Ta[:3, :3], Ta[:3, 3]
    Rb, tb = Tb[:3, :3], Tb[:3, 3]
    dR = Ra @ Rb.T
    dang = _rot_angle(dR)
    dtrans = np.linalg.norm(ta - tb)
    return float(w_rot * dang + w_trans * dtrans)

def robust_se3_average(T_list, max_iters=5, k_mad=2.5, return_stats=False):
    """
    Robust mean of SE3 via:
      - initial mean (quat Markley + trans mean)
      - iteratively reject outliers using MAD on residuals

    Returns:
      T_mean
      (optional) stats dict
    """
    if len(T_list) == 0:
        raise ValueError("T_list is empty")

    Ts = [np.asarray(T, dtype=np.float64) for T in T_list]

    def compute_mean(Ts_in):
        quats = []
        trans = []
        for T in Ts_in:
            q = _R_to_quat(T[:3, :3])
            quats.append(q)
            trans.append(T[:3, 3])
        quats = np.asarray(quats, dtype=np.float64)
        trans = np.asarray(trans, dtype=np.float64)

        # align quat signs
        q0 = quats[0]
        for i in range(len(quats)):
            if np.dot(quats[i], q0) < 0:
                quats[i] = -quats[i]

        w = np.ones((len(quats),), dtype=np.float64)
        q_mean = _average_quats(quats, w)
        t_mean = np.mean(trans, axis=0)

        Tm = np.eye(4, dtype=np.float64)
        Tm[:3, :3] = _quat_to_R(q_mean)
        Tm[:3, 3] = t_mean
        return Tm

    T_mean = compute_mean(Ts)
    inlier_mask = np.ones(len(Ts), dtype=bool)

    for _ in range(max_iters):
        res = np.array(
            [se3_distance(T, T_mean, w_rot=1.0, w_trans=1.0) for T in Ts],
            dtype=np.float64
        )
        med = np.median(res)
        mad = np.median(np.abs(res - med)) + 1e-12
        thr = med + k_mad * 1.4826 * mad

        new_mask = res <= thr
        if new_mask.sum() < max(3, int(0.3 * len(Ts))):
            break

        inlier_mask = new_mask
        T_new = compute_mean([T for T, m in zip(Ts, inlier_mask) if m])
        if se3_distance(T_new, T_mean) < 1e-6:
            T_mean = T_new
            break
        T_mean = T_new

    # -------- stats --------
    # rotation deviation (deg), translation deviation (mm)
    rot_devs = []
    trans_devs = []
    for T in Ts:
        dR = T[:3, :3] @ T_mean[:3, :3].T
        rot_devs.append(_rot_angle(dR) * 180.0 / np.pi)
        trans_devs.append(np.linalg.norm(T[:3, 3] - T_mean[:3, 3]) * 1000.0)

    stats = {
        "num_frames": len(Ts),
        "num_inliers": int(inlier_mask.sum()),
        "inlier_ratio": float(inlier_mask.mean()),
        "rotation_std_deg": float(np.std(rot_devs)),
        "rotation_mean_deg": float(np.mean(rot_devs)),
        "translation_std_mm": float(np.std(trans_devs)),
        "translation_mean_mm": float(np.mean(trans_devs)),
    }

    if return_stats:
        return T_mean, stats
    return T_mean
