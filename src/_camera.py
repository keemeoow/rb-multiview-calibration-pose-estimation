# camera.py 
import threading
import time
from typing import Optional, Tuple, Dict

import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    def __init__(
        self,
        serial: str,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        use_color: bool = True,
        use_depth: bool = False,
        align_depth_to_color: bool = True,
        warmup_frames: int = 10,
        frame_timeout_ms: int = 2000,
        warmup_timeout_ms: int = 2000,
        log_timeouts: bool = False,
        log_errors: bool = False,
        log_throttle_sec: float = 2.0,
        start_retries: int = 2,
        reset_on_start_failure: bool = True,
        startup_settle_sec: float = 1.0,
    ):
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.use_color = bool(use_color)
        self.use_depth = bool(use_depth)
        self.align_depth_to_color = bool(align_depth_to_color)
        self.warmup_frames = int(warmup_frames)
        self.frame_timeout_ms = int(frame_timeout_ms)
        self.warmup_timeout_ms = int(warmup_timeout_ms)
        self.log_timeouts = bool(log_timeouts)
        self.log_errors = bool(log_errors)
        self.log_throttle_sec = float(log_throttle_sec)
        self.start_retries = int(start_retries)
        self.reset_on_start_failure = bool(reset_on_start_failure)
        self.startup_settle_sec = float(startup_settle_sec)

        self.pipeline = None
        self.cfg = None
        self._build_pipeline_config()

        self.align = rs.align(rs.stream.color) if (self.use_depth and self.align_depth_to_color and self.use_color) else None

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._color = None
        self._depth = None
        self._ts_ms = None

        # Stream health counters (diagnostics)
        self._frames_received = 0
        self._wait_timeouts = 0
        self._loop_errors = 0
        self._stale_frames = 0
        self._last_error_msg = None
        self._last_error_wall_ms = None
        self._last_log_mono = 0.0

    def _build_pipeline_config(self):
        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial)

        if self.use_color:
            self.cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.use_depth:
            self.cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

    @staticmethod
    def list_devices() -> Dict[str, str]:
        ctx = rs.context()
        out = {}
        for dev in ctx.query_devices():
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            out[serial] = name
        return out

    @staticmethod
    def probe_device_info(serial: str) -> Dict[str, str]:
        ctx = rs.context()
        for dev in ctx.query_devices():
            s = dev.get_info(rs.camera_info.serial_number)
            if s != str(serial):
                continue
            info = {
                "serial": s,
                "name": dev.get_info(rs.camera_info.name),
            }
            try:
                info["usb_type"] = dev.get_info(rs.camera_info.usb_type_descriptor)
            except Exception:
                info["usb_type"] = "unknown"
            try:
                info["product_line"] = dev.get_info(rs.camera_info.product_line)
            except Exception:
                info["product_line"] = "unknown"
            return info
        return {"serial": str(serial), "name": "unknown", "usb_type": "unknown", "product_line": "unknown"}

    def _hardware_reset(self) -> bool:
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                if dev.get_info(rs.camera_info.serial_number) == self.serial:
                    dev.hardware_reset()
                    return True
        except Exception:
            pass
        return False

    def _warmup(self):
        arrived = 0
        max_attempts = max(self.warmup_frames * 3, self.warmup_frames + 2)
        for attempt in range(max_attempts):
            try:
                self.pipeline.wait_for_frames(timeout_ms=self.warmup_timeout_ms)
                arrived += 1
                if arrived >= self.warmup_frames:
                    return
            except Exception as e:
                print(f"[WARN] serial={self.serial} warmup {attempt+1}/{max_attempts}: {e}")
        raise RuntimeError(
            f"Camera serial={self.serial}: warmup failed "
            f"(requested {self.warmup_frames} frames, got {arrived})."
        )

    def start(self):
        last_exc = None
        n_tries = max(1, self.start_retries + 1)
        dev_info = self.probe_device_info(self.serial)
        print(
            f"[INFO] starting serial={self.serial} "
            f"name={dev_info.get('name','?')} usb={dev_info.get('usb_type','unknown')}"
        )

        for start_try in range(1, n_tries + 1):
            try:
                # Previous failure may have invalidated the pipeline/device handle.
                self._build_pipeline_config()
                self.pipeline.start(self.cfg)
                if self.startup_settle_sec > 0:
                    time.sleep(self.startup_settle_sec)
                self._warmup()
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                print(f"[WARN] serial={self.serial} start attempt {start_try}/{n_tries} failed: {e}")
                try:
                    self.pipeline.stop()
                except Exception:
                    pass

                if start_try < n_tries and self.reset_on_start_failure:
                    did_reset = self._hardware_reset()
                    print(
                        f"[INFO] serial={self.serial} hardware_reset "
                        f"{'issued' if did_reset else 'skipped'} before retry"
                    )
                    # Device re-enumeration after hardware reset takes time.
                    time.sleep(3.0 if did_reset else 1.0)
                elif start_try < n_tries:
                    time.sleep(1.0)

        if last_exc is not None:
            raise RuntimeError(
                f"Camera serial={self.serial}: start/warmup failed after {n_tries} attempts. "
                f"usb={dev_info.get('usb_type','unknown')}, depth={self.use_depth}, "
                f"profile={self.width}x{self.height}@{self.fps}. Last error: {last_exc}"
            ) from last_exc

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def _loop(self):
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=self.frame_timeout_ms)
                if self.align is not None:
                    frames = self.align.process(frames)

                color = frames.get_color_frame() if self.use_color else None
                depth = frames.get_depth_frame() if self.use_depth else None

                if self.use_color and (color is None):
                    continue

                ts_ms = None
                if color is not None:
                    ts_ms = float(color.get_timestamp())
                elif depth is not None:
                    ts_ms = float(depth.get_timestamp())

                with self._lock:
                    prev_ts_ms = self._ts_ms
                    if color is not None:
                        self._color = np.asanyarray(color.get_data()).copy()
                    if depth is not None:
                        self._depth = np.asanyarray(depth.get_data()).copy()
                    self._ts_ms = ts_ms
                    self._frames_received += 1
                    if (prev_ts_ms is not None) and (ts_ms is not None) and (float(prev_ts_ms) == float(ts_ms)):
                        self._stale_frames += 1

            except Exception as e:
                msg = str(e)
                is_timeout = ("didn't arrive within" in msg) or ("timeout" in msg.lower())
                should_log = False
                log_text = ""
                now_mono = time.monotonic()

                with self._lock:
                    if is_timeout:
                        self._wait_timeouts += 1
                    else:
                        self._loop_errors += 1
                    self._last_error_msg = f"{type(e).__name__}: {msg}"
                    self._last_error_wall_ms = time.time() * 1000.0

                    want_log = (is_timeout and self.log_timeouts) or ((not is_timeout) and self.log_errors)
                    if want_log and ((now_mono - self._last_log_mono) >= self.log_throttle_sec):
                        self._last_log_mono = now_mono
                        should_log = True
                        log_text = (
                            f"[RS][WARN] serial={self.serial} "
                            f"{'timeout' if is_timeout else 'error'}: {type(e).__name__}: {msg} "
                            f"(frames={self._frames_received}, timeouts={self._wait_timeouts}, "
                            f"errors={self._loop_errors}, stale={self._stale_frames})"
                        )

                if should_log:
                    print(log_text)
                time.sleep(0.005)

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        with self._lock:
            c = None if self._color is None else self._color.copy()
            d = None if self._depth is None else self._depth.copy()
            ts = self._ts_ms
        return c, d, ts

    def get_stats(self) -> Dict[str, object]:
        with self._lock:
            return {
                "serial": self.serial,
                "frames_received": int(self._frames_received),
                "wait_timeouts": int(self._wait_timeouts),
                "loop_errors": int(self._loop_errors),
                "stale_frames": int(self._stale_frames),
                "last_ts_ms": (None if self._ts_ms is None else float(self._ts_ms)),
                "last_error_msg": self._last_error_msg,
                "last_error_wall_ms": self._last_error_wall_ms,
            }
