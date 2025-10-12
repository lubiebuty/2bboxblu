#!/usr/bin/env python3
"""
Śledzenie markera ArUco z łańcuchem odporności:
ArUco (pełny kadr) → re-detect w ROI → LK (4 narożniki) → CSRT.
Ulepszenia:
- undistort przed detekcją (newK + remap),
- subpikselowe narożniki (corner refinement),
- środek markera przez projectPoints((0,0,0)) z pozy (jeśli kalibracja),
- spójne rysowanie/zapis na obrazie po undistort.

Wyjścia:
- positions_px.csv (t, cx, cy, z_cm, dy_px)
- deflection_px.png (+ deflection_z_cm.png jeśli mamy kalibrację)
"""
from __future__ import annotations
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys, time, os, re
from typing import List, Tuple, Dict, Any

from datetime import datetime
import shutil

# ------------ Ustawienia -------------
# ------------ Ustawienia -------------
VIDEO_PATH = "/Users/bartlomiejostasz/PYCH/nagrania/12 pazdziernik niedziela rano /test long.MOV"
# plik kalibracyjny kamery (macierz K i dystorsja)
CALIB_PATH = '/Users/bartlomiejostasz/PYCH/nagrania/dane do kalibracji /charuco_calibration_with_distance.npz'
ARUCO_DICT = "DICT_6X6_1000"   # ręcznie lub AUTO w przyszłości
MARKER_ID  = 2                 # None = pierwszy wykryty
OUT_VIDEO  = "out_annotated.mp4"
# numer serii; jeśli None, wybierze się automatycznie kolejny
SERIES_NO = None  # numer serii; jeśli None, wybierze się automatycznie kolejny
# MARKER_LEN_M = 0.034           # wymiar boku markera w metrach (tu: 3.4 cm)
MARKER_LEN_M = 0.034           # wymiar boku markera w metrach (tu: 3.4 cm)
# --- Skalowanie px→cm dla osi pionowej ---
# Jeśli ustawisz stałą odległość (np. 3.0 m), wykres [cm] zachowa kształt [px].
LOCK_PLANE_DISTANCE_M = 3.0   # None = brak blokady; licz z PnP. 3.0 = zawsze 3 m.
CONST_SCALE_FROM_FIRST_N = 0   # Jeśli >0 i LOCK=None: po N klatkach z PnP policz medianę Z i zablokuj skalę
# -------------------------------------

# słowniki OpenCV
ARUCO_DICTS: Dict[str, int] = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco) if name.startswith("DICT_")
}

# --- Helpers: numer serii i folder wyjściowy ---

def _next_series_number(base: Path) -> int:
    """Zwraca kolejny numer serii na podstawie istniejących folderów 'seria N - ...' w katalogu programu."""
    max_no = 0
    for p in base.iterdir():
        if p.is_dir():
            m = re.match(r"(?i)^seria[ _-]?(\d+)", p.name)
            if m:
                try:
                    n = int(m.group(1))
                    if n > max_no:
                        max_no = n
                except Exception:
                    pass
    return max_no + 1


def _make_output_dir(series_no: int | None) -> tuple[Path, int]:
    """Tworzy folder 'seria {N} - YYYY-MM-DD_HH-MM-SS' w folderze programu i zwraca (sciezka, N)."""
    base = Path(__file__).resolve().parent
    if series_no is None:
        series_no = _next_series_number(base)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"seria {series_no} - {ts}"
    out_dir = base / folder_name
    # unikaj kolizji nazw (mało prawdopodobne, ale na wszelki wypadek)
    idx = 2
    while out_dir.exists():
        out_dir = base / f"{folder_name}_{idx}"
        idx += 1
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir, series_no


class ArucoTracker:
    def __init__(
                 self,
                 video_path: str | int,
                 dict_name: str,
                 marker_id: int | None,
                 out_path: str,
                 camera_matrix=None,
                 dist_coeffs=None,
                 out_dir: str | Path = ".") -> None:
        if dict_name not in ARUCO_DICTS:
            raise ValueError("Nieznany słownik ArUco")
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            sys.exit("❌ Nie można otworzyć wideo")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.w  = int(self.cap.get(3))
        self.h  = int(self.cap.get(4))

        # --- ArUco dictionary + detector ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
        if hasattr(cv2.aruco, "ArucoDetector"):
            try:
                params = cv2.aruco.DetectorParameters()
            except AttributeError:
                params = cv2.aruco.DetectorParameters_create()
            # Tuning detektora
            params.adaptiveThreshWinSizeMin = 5
            params.adaptiveThreshWinSizeMax = 35
            params.adaptiveThreshWinSizeStep = 5
            params.minMarkerPerimeterRate = 0.02
            params.minDistanceToBorder = 5
            if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            elif hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
            self._detect = lambda img: self.detector.detectMarkers(img)
        else:
            try:
                self.parameters = cv2.aruco.DetectorParameters_create()
            except AttributeError:
                self.parameters = cv2.aruco.DetectorParameters()
            self.parameters.adaptiveThreshWinSizeMin = 5
            self.parameters.adaptiveThreshWinSizeMax = 35
            self.parameters.adaptiveThreshWinSizeStep = 5
            self.parameters.minMarkerPerimeterRate = 0.02
            self.parameters.minDistanceToBorder = 5
            if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
                self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            elif hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG"):
                self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
            self._detect = lambda img: cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=self.parameters)

        self.marker_id = marker_id

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, self.fps, (self.w, self.h))

        try:
            self.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            self.tracker = cv2.legacy.TrackerCSRT_create()

        # Stany główne
        self.track_mode = False        # False = ArUco, True = CSRT
        self.last_bbox = None
        self.csrt_ready = False
        self.lost_counter = 0
        self.lost_thresh  = 1

        self.records: List[Dict[str, Any]] = []
        self.first_center: Tuple[float,float] | None = None

        self.K = camera_matrix
        self.D = dist_coeffs

        # katalog wyjściowy
        self.out_dir = Path(out_dir)

        # --- Precompute undistort maps ---
        self.use_undistort = False
        if self.K is not None and self.D is not None:
            # alpha=0 → minimalne czarne brzegi, pełne dopasowanie do (w,h)
            self.newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, (self.w, self.h), 0)
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K, self.D, None, self.newK, (self.w, self.h), cv2.CV_16SC2
            )
            # zerowa dystorsja dla newK
            self.zeroD = np.zeros((1,5), dtype=np.float32)
            self.use_undistort = True

        # --- PnP / ray–plane pomocnicze stany ---
        self.last_rvec = None   # 3x1, ostatnia dobra poza (ArUco)
        self.last_tvec = None   # 3x1
        self.last_z_m = None    # ostatnie Z w metrach
        self.first_marker_xy_cm = None  # (X0_cm, Y0_cm) – odniesienie w ukł. markera

        # --- Stały układ odniesienia płaszczyzny (z pierwszej dobrej pozy) ---
        self.ref_R = None  # 3x3 macierz R z pierwszej dobrej klatki
        self.ref_t = None  # 3x1 wektor t z pierwszej dobrej klatki
        # Precompute fx, fy dla prostego przelicznika px→cm (fallback)
        if self.use_undistort:
            self.fx = float(self.newK[0, 0]); self.fy = float(self.newK[1, 1])
        else:
            self.fx = float(self.K[0, 0]) if self.K is not None else None
            self.fy = float(self.K[1, 1]) if self.K is not None else None

        # --- Skala pionowa px→cm ---
        self.scale_y_cm_per_px: float | None = None
        self.scale_locked = False
        self._z_samples: list[float] = []
        if self.fy is not None and isinstance(LOCK_PLANE_DISTANCE_M, (int, float)) and LOCK_PLANE_DISTANCE_M is not None:
            self.scale_y_cm_per_px = (float(LOCK_PLANE_DISTANCE_M) * 100.0) / self.fy
            self.scale_locked = True

        # --- Lucas–Kanade (LK) fallback i ROI re-detect ---
        self.lk_active = False
        self.prev_gray = None
        self.prev_pts = None
        self.lk_winSize = (15, 15)
        self.lk_maxLevel = 3
        self.lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        self.roi_expand = 1.6  # przy re-detect powiększamy ostatni bbox

        # --- BLUE ROI (łańcuch) + bezpieczeństwo ---
        self.blue_roi = None                    # (x,y,w,h) – ROI do SZUKANIA (tylko po ArUco)
        self.blue_vis = None                    # (x,y,w,h) – box do PODGLĄDU (LK/CSRT/ekstrap.)
        self.blue_roi_half = 40                 # stałe ±40 px od środka (podstawowe okno)
        self.blue_roi_half_big = 160            # „winda bezpieczeństwa” po 1 missie (większa)
        self.blue_miss = 0                      # kolejne missy w BLUE ROI
        self.anchor_every = 5                   # co N klatek próba pełnokadrowa (częściej)

        # --- LK gating & pamięć pewnych pomiarów ---
        self.lk_fb_thresh = 1.5                 # maks. błąd forward-backward [px]
        self.lk_max_step = int(0.75 * self.blue_roi_half_big)  # maks. krok punktu/klatkę [px]
        self.lk_bad_counter = 0                 # kolejne klatki z niewiarygodnym LK
        self.last_good_c = None                 # ostatni wiarygodny czworokąt (4x2)
        self.last_good_center = None            # i jego środek (cx, cy)

    def _detect_in_roi(self, frame: np.ndarray, bbox: Tuple[int,int,int,int], id_wanted: int | None):
        """Wykryj ArUco w powiększonym ROI wokół ostatniego bboxa.
        Zwraca: (corners_4x2, id) lub (None, None).
        """
        x, y, w, h = bbox
        pad_x = int(w * (self.roi_expand - 1.0) / 2.0)
        pad_y = int(h * (self.roi_expand - 1.0) / 2.0)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(self.w, x + w + pad_x)
        y2 = min(self.h, y + h + pad_y)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None
        roi_p = self._preprocess_roi(roi)
        corners, ids, _ = self._detect(roi_p)
        if ids is None or not len(ids):
            return None, None
        ids = ids.flatten()
        idx = 0
        if id_wanted is not None:
            matches = np.where(ids == id_wanted)[0]
            if len(matches) == 0:
                return None, None
            idx = int(matches[0])
        c = corners[idx][0].copy()  # 4x2
        c[:, 0] += x1
        c[:, 1] += y1
        return c, int(ids[idx])

    @staticmethod
    def _clip_bbox(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int,int,int,int]:
        x = max(0, x); y = max(0, y)
        w = max(2, min(W - x, w)); h = max(2, min(H - y, h))
        return (int(x), int(y), int(w), int(h))

    def _preprocess_roi(self, gray_roi: np.ndarray) -> np.ndarray:
        """Wstępne przetwarzanie ROI w skali szarości: CLAHE + lekkie wyostrzenie.
        Pomaga przy blikach wody i słabszym kontraście."""
        if gray_roi is None or gray_roi.size == 0:
            return gray_roi
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(gray_roi)
        blur = cv2.GaussianBlur(g, (3, 3), 0)
        sharp = cv2.addWeighted(g, 1.5, blur, -0.5, 0)
        return sharp

    def _detect_multiscale(self, gray: np.ndarray, scales=(1.0, 1.25, 0.75)):
        """Detekcja pełnokadrowa w kilku skalach. Zwraca (corners_4x2, id) albo (None, None)."""
        for s in scales:
            if s == 1.0:
                img = gray
            else:
                interp = cv2.INTER_LINEAR if s > 1.0 else cv2.INTER_AREA
                img = cv2.resize(gray, None, fx=s, fy=s, interpolation=interp)
            corners, ids, _ = self._detect(img)
            if ids is None or not len(ids):
                continue
            ids = ids.flatten()
            idx = 0
            if self.marker_id is not None:
                matches = np.where(ids == self.marker_id)[0]
                if len(matches) == 0:
                    continue
                idx = int(matches[0])
            c = corners[idx][0].copy()
            if s != 1.0:
                c /= s
            return c, int(ids[idx])
        return None, None

    def _lk_confidence(self, c_lk: np.ndarray, last_bbox: Tuple[int,int,int,int] | None) -> bool:
        """Oceń czy kształt z LK wygląda jak kwadrat markera (w przybliżeniu)."""
        if c_lk is None or len(c_lk) != 4:
            return False
        try:
            if not cv2.isContourConvex(c_lk.astype(np.float32)):
                return False
        except Exception:
            return False
        d = [float(np.linalg.norm(c_lk[(i+1) % 4] - c_lk[i])) for i in range(4)]
        mn = max(1e-6, min(d)); mx = max(d)
        if mx / mn > 2.2:  # zbyt wydłużony
            return False
        area = float(cv2.contourArea(c_lk.astype(np.float32)))
        if last_bbox is not None:
            area_bbox = float(max(1, last_bbox[2] * last_bbox[3]))
            ratio = area / area_bbox
            if ratio < 0.25 or ratio > 4.0:
                return False
        return True

    def _lk_fb_filter(self, prev_gray: np.ndarray, gray: np.ndarray,
                      prev_pts: np.ndarray, next_pts: np.ndarray, st: np.ndarray):
        """Forward–backward check + gating kroku i ROI.
        Zwraca (next_pts_filtered, valid_mask) lub (None, None) jeśli za mało punktów.
        """
        if next_pts is None or st is None or prev_pts is None or prev_gray is None:
            return None, None
        st_f = (st.flatten() == 1)
        if not np.any(st_f):
            return None, None
        # Oblicz ruch wsteczny
        back_pts, st_b, _ = cv2.calcOpticalFlowPyrLK(
            gray, prev_gray, next_pts, None,
            winSize=self.lk_winSize, maxLevel=self.lk_maxLevel, criteria=self.lk_criteria
        )
        if back_pts is None:
            return None, None
        prev_flat = prev_pts.reshape(-1, 2)
        next_flat = next_pts.reshape(-1, 2)
        back_flat = back_pts.reshape(-1, 2)
        fb_err = np.linalg.norm(prev_flat - back_flat, axis=1)
        step = np.linalg.norm(next_flat - prev_flat, axis=1)
        valid = st_f & (fb_err < self.lk_fb_thresh) & (step < self.lk_max_step)
        # Gating do rozszerzonego ROI (jeśli mamy)
        if self.blue_roi is not None:
            bx, by, bw, bh = self.blue_roi
            # rozszerz o 25%
            ex = int(0.125 * bw); ey = int(0.125 * bh)
            gx1, gy1 = max(0, bx - ex), max(0, by - ey)
            gx2, gy2 = min(self.w - 1, bx + bw + ex), min(self.h - 1, by + bh + ey)
            inside = (
                (next_flat[:, 0] >= gx1) & (next_flat[:, 0] <= gx2) &
                (next_flat[:, 1] >= gy1) & (next_flat[:, 1] <= gy2)
            )
            valid = valid & inside
        if int(valid.sum()) < 3:
            return None, None
        return next_pts[valid].reshape(-1, 1, 2), valid

    def _detect_in_bbox(self, frame: np.ndarray, bbox: Tuple[int,int,int,int], id_wanted: int | None):
        """Wykryj ArUco w *dokładnym* bboxie (bez dodatkowego skalowania)."""
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None, None
        roi_p = self._preprocess_roi(roi)
        corners, ids, _ = self._detect(roi_p)
        if ids is None or not len(ids):
            return None, None
        ids = ids.flatten()
        idx = 0
        if id_wanted is not None:
            matches = np.where(ids == id_wanted)[0]
            if len(matches) == 0:
                return None, None
            idx = int(matches[0])
        c = corners[idx][0].copy()
        c[:,0] += x
        c[:,1] += y
        return c, int(ids[idx])

    def _pixel_to_marker_cm(self, u: float, v: float, rvec: np.ndarray, tvec: np.ndarray) -> Tuple[float, float]:
        """Rzutuj piksel (u,v) na płaszczyznę markera i zwróć (X_cm, Y_cm) w układzie markera.
        Zakładamy, że obraz jest po undistort i używamy macierzy self.newK, a dystorsja = 0.
        Jeśli nie mamy newK (brak kalibracji), używamy self.K jako przybliżenia.
        """
        K = self.newK if hasattr(self, 'newK') and self.newK is not None else self.K
        if K is None:
            # Bez kalibracji nie zrobimy poprawnego ray–plane → wróć 0,0 (wyżej zabezpieczamy wywołanie)
            return 0.0, 0.0
        if rvec is None or tvec is None:
            return 0.0, 0.0
        R, _ = cv2.Rodrigues(rvec.reshape(3,1))
        t = tvec.reshape(3,1)
        n = R @ np.array([[0.0],[0.0],[1.0]], dtype=np.float64)  # normalna płaszczyzny markera w ukł. kamery
        dinv = np.linalg.inv(K) @ np.array([[float(u)],[float(v)],[1.0]], dtype=np.float64)  # kierunek promienia
        denom_arr = (n.T @ dinv)
        denom = float(denom_arr.item())
        if abs(denom) < 1e-9:
            # promień równoległy do płaszczyzny – numerycznie niepewne
            return 0.0, 0.0
        num = float((n.T @ t).item())
        lam = num / denom
        X_cam = lam * dinv                      # 3x1 punkt przecięcia w kamerze
        X_mark = R.T @ (X_cam - t)              # do układu markera
        X_cm = float(X_mark[0, 0] * 100.0)
        Y_cm = float(X_mark[1, 0] * 100.0)
        return X_cm, Y_cm

    def _cm_from_pose(self, rvec: np.ndarray, tvec: np.ndarray) -> Tuple[float, float] | Tuple[None, None]:
        """Policz (dx_cm, dy_cm) jako rzut różnicy położeń środka markera na osie płaszczyzny
        z pierwszej dobrej klatki: e1=R0[:,0], e2=R0[:,1]. Jednostki: cm.
        Zwraca (None, None) jeśli brak referencji.
        """
        if rvec is None or tvec is None or self.ref_R is None or self.ref_t is None:
            return None, None
        R, _ = cv2.Rodrigues(rvec.reshape(3,1))
        t = tvec.reshape(3,1)
        d = t - self.ref_t  # 3x1 w kamerze
        e1 = self.ref_R[:, [0]]  # 3x1
        e2 = self.ref_R[:, [1]]  # 3x1
        dx = float((d.T @ e1).item()) * 100.0
        dy = float((d.T @ e2).item()) * 100.0
        return dx, dy

    def run(self) -> None:
        frame_idx = 0
        start = time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            t = frame_idx / self.fps

            # Undistort (jeśli dostępne) + gray
            frame_vis = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR) if self.use_undistort else frame
            gray = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2GRAY)

            detected = False
            c = None  # aktualny czworokąt 4x2
            z_cm = None
            used_blue = False
            blue_updated = False
            search_roi_moved = False

            # --- ŁAŃCUCHOWE WYSZUKIWANIE ---
            if self.blue_roi is not None:
                # 1) Szukaj w niebieskim b-boxie z poprzedniej klatki
                c_roi, _ = self._detect_in_bbox(gray, self.blue_roi, self.marker_id)
                if c_roi is not None:
                    c = c_roi; detected = True; used_blue = True; self.blue_miss = 0
                else:
                    # 1a) Winda bezpieczeństwa: powiększ okno na tę jedną próbę
                    cx_b = self.blue_roi[0] + self.blue_roi[2]/2.0
                    cy_b = self.blue_roi[1] + self.blue_roi[3]/2.0
                    half = self.blue_roi_half_big
                    bx = int(round(cx_b - half)); by = int(round(cy_b - half))
                    bw = int(2*half); bh = int(2*half)
                    bx, by, bw, bh = self._clip_bbox(bx, by, bw, bh, self.w, self.h)
                    big_box = (bx, by, bw, bh)
                    self.blue_vis = big_box
                    c_big, _ = self._detect_in_bbox(gray, big_box, self.marker_id)
                    if c_big is not None:
                        c = c_big; detected = True; used_blue = True; self.blue_miss = 0
                    else:
                        self.blue_miss += 1
                        # 1b) Po 1 missie – kotwica pełnokadrowa (szybszy powrót)
                        if self.blue_miss >= 1:
                            c_ff, id_ff = self._detect_multiscale(gray)
                            if c_ff is not None:
                                c = c_ff; detected = True; used_blue = False; self.blue_miss = 0
            else:
                # 0) Pierwsza kotwica: pełny kadr (multiskala)
                c_full, id_full = self._detect_multiscale(gray)
                if c_full is not None:
                    c = c_full; detected = True

            # 0b) Dodatkowa kotwica co N klatek (gdy korzystaliśmy z BLUE ROI)
            if detected and used_blue and self.anchor_every and (frame_idx % self.anchor_every == 0):
                c_a, id_a = self._detect_multiscale(gray)
                if c_a is not None:
                    c = c_a

            if detected and c is not None:
                self.lost_counter = 0

                # Środek markera: z geometrii jeśli mamy kalibrację, inaczej centroid narożników
                if self.use_undistort:
                    # OpenCV oczekuje kształtu (N,1,4,2); robimy 1 marker
                    corners_ = c.reshape(1, 1, 4, 2).astype(np.float32)
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners_, MARKER_LEN_M, self.newK, self.zeroD)
                    rvec0 = rvec[0].reshape(3,1)
                    tvec0 = tvec[0].reshape(3,1)
                    center_img, _ = cv2.projectPoints(np.array([[0., 0., 0.]], np.float32), rvec0, tvec0, self.newK, self.zeroD)
                    cx, cy = map(float, center_img.ravel())
                    z_cm = float(tvec0[2,0] * 100.0)
                    # zapamiętaj ostatnią dobrą pozę do forward-fill
                    self.last_rvec, self.last_tvec = rvec0.copy(), tvec0.copy()
                    self.last_z_m = float(tvec0[2,0])
                    # Auto-blokada skali z mediany Z z pierwszych N klatek (jeśli włączono)
                    if (not self.scale_locked) and self.fy is not None and LOCK_PLANE_DISTANCE_M is None and CONST_SCALE_FROM_FIRST_N > 0:
                        self._z_samples.append(self.last_z_m)
                        if len(self._z_samples) >= CONST_SCALE_FROM_FIRST_N:
                            z_med = float(np.median(self._z_samples))
                            self.scale_y_cm_per_px = (z_med * 100.0) / self.fy
                            self.scale_locked = True
                            print(f"🔒 Zablokowano skalę Δy: {self.scale_y_cm_per_px:.6f} cm/px (Z_med={z_med:.3f} m)")
                    # ustaw referencję płaszczyzny przy pierwszej dobrej pozie
                    if self.ref_R is None or self.ref_t is None:
                        self.ref_R, _ = cv2.Rodrigues(rvec0)
                        self.ref_t = tvec0.copy()
                    # przemieszczenia w cm względem referencji (oś płaszczyzny z pierwszej klatki)
                    dx_pose_cm, dy_pose_cm = self._cm_from_pose(rvec0, tvec0)
                    x_cm = None; y_cm = None
                else:
                    cx, cy = c.mean(axis=0)
                    z_cm = None
                    dx_pose_cm = None; dy_pose_cm = None
                    x_cm = None; y_cm = None

                # bbox z paddingiem
                pad = 15
                x_min, y_min = np.min(c, axis=0)
                x_max, y_max = np.max(c, axis=0)
                x_min = max(0, int(x_min) - pad)
                y_min = max(0, int(y_min) - pad)
                x_max = min(self.w - 1, int(x_max) + pad)
                y_max = min(self.h - 1, int(y_max) + pad)
                bbox = (
                    x_min,
                    y_min,
                    max(2, x_max - x_min),
                    max(2, y_max - y_min),
                )
                self.last_bbox = bbox

                # Re-init CSRT wokół bieżącego bboxa (na wypadek dalszego fallbacku)
                if self.track_mode or not self.csrt_ready:
                    try:
                        self.tracker = cv2.TrackerCSRT_create()
                    except AttributeError:
                        self.tracker = cv2.legacy.TrackerCSRT_create()
                    self.tracker.init(frame_vis, bbox)
                    self.csrt_ready = True
                self.track_mode = False

                # Init LK (4 narożniki)
                self.lk_active = True
                self.prev_gray = gray.copy()
                self.prev_pts = c.reshape(-1, 1, 2).astype(np.float32)

                if self.first_center is None:
                    self.first_center = (cx, cy)
                # rysowanie
                cv2.polylines(frame_vis, [c.astype(int)], True, (0, 255, 0), 2)
                cv2.circle(frame_vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                # przemieszczenia w px (zgodnie z dotychczasową konwencją dla Y)
                dx_px = cx - self.first_center[0]
                dy_px = self.first_center[1] - cy
                # przemieszczenie pionowe w cm: preferuj stałą skalę (LOCK), inaczej bieżące Z
                if self.scale_y_cm_per_px is not None:
                    dy_cm = dy_px * self.scale_y_cm_per_px
                elif self.last_z_m is not None and self.fy is not None:
                    dy_cm = (dy_px / self.fy) * self.last_z_m * 100.0
                else:
                    dy_cm = None
                dx_cm = None  # nie wykorzystujemy poziomego cm w tym wariancie
                # zapis rekordu jako słownik (łatwiejsze rozszerzanie)
                self.records.append({
                    "t": t, "cx": cx, "cy": cy, "z_cm": z_cm,
                    "x_cm": x_cm, "y_cm": y_cm,
                    "dx_px": dx_px, "dy_px": dy_px,
                    "dx_cm": dx_cm, "dy_cm": dy_cm,
                    "source": "aruco"
                })

                # --- AKTUALIZUJ BLUE ROI (po ArUco) ---
                half = self.blue_roi_half
                bx = int(round(cx - half)); by = int(round(cy - half))
                bw = int(2*half); bh = int(2*half)
                self.blue_roi = self._clip_bbox(bx, by, bw, bh, self.w, self.h)   # ← TYLKO po udanym ArUco
                self.blue_vis = self.blue_roi                                      # podgląd = to samo
                search_roi_moved = True
                blue_updated = True

            else:
                # Brak ArUco: 3) LK narożników, 4) CSRT
                self.lost_counter += 1
                used_lk = False

                if self.lk_active and self.prev_gray is not None and self.prev_pts is not None:
                    next_pts, st, err = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, self.prev_pts, None,
                        winSize=self.lk_winSize, maxLevel=self.lk_maxLevel, criteria=self.lk_criteria
                    )
                    if next_pts is not None and st is not None:
                        # Forward–backward + kroki + ROI gating
                        next_f, valid_mask = self._lk_fb_filter(self.prev_gray, gray, self.prev_pts, next_pts, st)
                        if next_f is not None:
                            # Złóż pełny zestaw 4 punktów: aktualizuj tylko te, które przeszły filtr
                            curr_pts = self.prev_pts.copy()
                            curr_pts[valid_mask] = next_f

                            # Jeśli brakuje jednego narożnika – odtwarzamy go afinicznie
                            if int(valid_mask.sum()) == 3:
                                src = self.prev_pts[valid_mask].reshape(3,2).astype(np.float32)
                                dst = next_f.reshape(3,2).astype(np.float32)
                                M = cv2.getAffineTransform(src, dst)
                                miss_idx = int(np.where(~valid_mask)[0][0])
                                p = self.prev_pts[miss_idx,0].astype(np.float32)
                                pred = (M @ np.array([p[0], p[1], 1.0], dtype=np.float32)).reshape(2)
                                curr_pts[miss_idx,0] = pred

                            c_lk = curr_pts.reshape(-1, 2)
                            conf = self._lk_confidence(c_lk, self.last_bbox)
                            if conf:
                                used_lk = True
                                cx = float(np.mean(c_lk[:, 0])); cy = float(np.mean(c_lk[:, 1]))

                                # Update bbox
                                x_min, y_min = np.min(c_lk, axis=0)
                                x_max, y_max = np.max(c_lk, axis=0)
                                pad = 15
                                x_min = max(0, int(x_min) - pad)
                                y_min = max(0, int(y_min) - pad)
                                x_max = min(self.w - 1, int(x_max) + pad)
                                y_max = min(self.h - 1, int(y_max) + pad)
                                self.last_bbox = (x_min, y_min, max(2, x_max - x_min), max(2, y_max - y_min))

                                # update stanu LK (pełny zestaw 4 punktów)
                                self.prev_gray = gray.copy()
                                self.prev_pts = curr_pts

                                # rysowanie tylko gdy wiarygodne
                                cv2.polylines(frame_vis, [c_lk.astype(int)], True, (0, 255, 255), 2)
                                cv2.circle(frame_vis, (int(cx), int(cy)), 4, (0, 255, 255), -1)
                                # przeliczenia w cm – fallback: lokalna skala z ostatniego Z i fx/fy
                                if self.first_center is None:
                                    self.first_center = (cx, cy)
                                dx_px = cx - self.first_center[0]
                                dy_px = self.first_center[1] - cy
                                if self.scale_y_cm_per_px is not None:
                                    dy_cm = dy_px * self.scale_y_cm_per_px
                                elif self.last_z_m is not None and self.fy is not None:
                                    dy_cm = (dy_px / self.fy) * self.last_z_m * 100.0
                                else:
                                    dy_cm = None
                                dx_cm = None
                                x_cm = None; y_cm = None
                                self.records.append({
                                    "t": t, "cx": cx, "cy": cy, "z_cm": None,
                                    "x_cm": x_cm, "y_cm": y_cm,
                                    "dx_px": dx_px, "dy_px": dy_px,
                                    "dx_cm": dx_cm, "dy_cm": dy_cm,
                                    "source": "lk"
                                })

                                # wizualny BLUE BOX
                                half = self.blue_roi_half
                                bx = int(round(cx - half)); by = int(round(cy - half))
                                bw = int(2*half); bh = int(2*half)
                                self.blue_vis = self._clip_bbox(bx, by, bw, bh, self.w, self.h)
                                blue_updated = True

                                # pozwól LK przesunąć także *search ROI*
                                half = self.blue_roi_half
                                bx = int(round(cx - half)); by = int(round(cy - half))
                                bw = int(2*half); bh = int(2*half)
                                self.blue_roi = self._clip_bbox(bx, by, bw, bh, self.w, self.h)
                                self.blue_vis = self.blue_roi
                                search_roi_moved = True
                                self.blue_miss = 0
                                self.lk_bad_counter = 0
                                self.last_good_c = c_lk.copy(); self.last_good_center = (cx, cy)
                            else:
                                # LK niewiarygodny: nie rysujemy igły, nie dopisujemy rekordów
                                self.lk_bad_counter += 1
                        else:
                            # Zbyt mało poprawnych punktów po filtracji
                            self.lk_active = False
                            self.prev_gray = None
                            self.prev_pts = None

                if not used_lk:
                    # CSRT fallback
                    if self.csrt_ready and self.lost_counter >= self.lost_thresh:
                        self.track_mode = True
                    if self.track_mode and self.csrt_ready:
                        ok_t, nbbox = self.tracker.update(frame_vis)
                        if ok_t:
                            cx = nbbox[0] + nbbox[2] / 2
                            cy = nbbox[1] + nbbox[3] / 2
                            scale = 1.20
                            nw, nh = int(nbbox[2] * scale), int(nbbox[3] * scale)
                            x = int(cx - nw / 2)
                            y = int(cy - nh / 2)
                            w, h = nw, nh
                            cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            cv2.circle(frame_vis, (int(cx), int(cy)), 4, (255, 0, 0), -1)
                            if self.first_center is None:
                                self.first_center = (cx, cy)
                            dx_px = cx - self.first_center[0]
                            dy_px = self.first_center[1] - cy
                            if self.scale_y_cm_per_px is not None:
                                dy_cm = dy_px * self.scale_y_cm_per_px
                            elif self.last_z_m is not None and self.fy is not None:
                                dy_cm = (dy_px / self.fy) * self.last_z_m * 100.0
                            else:
                                dy_cm = None
                            dx_cm = None
                            x_cm = None; y_cm = None
                            self.records.append({
                                "t": t, "cx": cx, "cy": cy, "z_cm": None,
                                "x_cm": x_cm, "y_cm": y_cm,
                                "dx_px": dx_px, "dy_px": dy_px,
                                "dx_cm": dx_cm, "dy_cm": dy_cm,
                                "source": "csrt"
                            })
                            # Aktualizuj TYLKO wizualny BLUE BOX (szukamy nadal w ostatnim ArUco ROI)
                            half = self.blue_roi_half
                            bx = int(round(cx - half)); by = int(round(cy - half))
                            bw = int(2*half); bh = int(2*half)
                            self.blue_vis = self._clip_bbox(bx, by, bw, bh, self.w, self.h)
                            blue_updated = True
                        else:
                            cv2.putText(frame_vis, "LOST", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame_vis, "LOST", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Jeśli w tej klatce nie udało się zaktualizować BLUE ROI (brak DETECT/LK/CSRT),
            # przesuń box wizualny ekstrapolacją z dwóch ostatnich zapisów (stała prędkość).
            if not blue_updated:
                if len(self.records) >= 2:
                    cx_prev, cy_prev = self.records[-2]["cx"], self.records[-2]["cy"]
                    cx_last, cy_last = self.records[-1]["cx"], self.records[-1]["cy"]
                    cx_pred = 2*cx_last - cx_prev
                    cy_pred = 2*cy_last - cy_prev
                    half = self.blue_roi_half
                    bx = int(round(cx_pred - half)); by = int(round(cy_pred - half))
                    bw = int(2*half); bh = int(2*half)
                    self.blue_vis = self._clip_bbox(bx, by, bw, bh, self.w, self.h)
                    blue_updated = True

            # Rysuj BLUE BOX: preferuj wizualny (może pochodzić z LK/CSRT/ekstrapolacji)
            box_to_draw = self.blue_vis if self.blue_vis is not None else self.blue_roi
            if box_to_draw is not None:
                x, y, w, h = box_to_draw
                cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

            self.writer.write(frame_vis)
            # tryb offline: brak podglądu na żywo
            frame_idx += 1

        self.cap.release(); self.writer.release(); cv2.destroyAllWindows()
        # zapisz wyniki i wykresy
        self._postprocess()

    def _postprocess(self) -> None:
        if not self.records:
            print("⚠ Brak zapisanych rekordów – pomijam CSV/wykresy.")
            return
        df = pd.DataFrame(self.records)
        # Uzupełnij dy_px jeśli brak
        if "dy_px" not in df.columns:
            if self.first_center is not None:
                df["dy_px"] = self.first_center[1] - df["cy"]
            else:
                df["dy_px"] = 0.0
        # Zapis CSV (pełny zestaw kolumn)
        cols_order = [c for c in [
            "t","cx","cy","z_cm","x_cm","y_cm","dx_px","dy_px","dx_cm","dy_cm","source"
        ] if c in df.columns]
        df.to_csv(self.out_dir / "positions_px.csv", index=False, columns=cols_order)
        # Wykres pionowy w pikselach
        plt.figure(figsize=(8,4))
        plt.plot(df["t"], df["dy_px"])
        plt.axhline(0, linewidth=0.8)
        plt.xlabel("Czas [s]"); plt.ylabel("Δy [px]")
        plt.title("Wychylenie pionowe [px]")
        plt.grid(True); plt.tight_layout()
        plt.savefig(self.out_dir / "deflection_y_px.png", dpi=150)
        plt.close()
        # Wykres pionowy w cm (jeśli dostępny)
        if "dy_cm" in df.columns and df["dy_cm"].notna().any():
            plt.figure(figsize=(8,4))
            plt.plot(df["t"], df["dy_cm"])
            plt.axhline(0, linewidth=0.8)
            plt.xlabel("Czas [s]"); plt.ylabel("Δy [cm]")
            plt.title("Wychylenie pionowe [cm]")
            plt.grid(True); plt.tight_layout()
            plt.savefig(self.out_dir / "deflection_y_cm.png", dpi=150)
            plt.close()
        saved = [self.out_dir / "positions_px.csv", self.out_dir / "deflection_y_px.png"]
        p_cm = self.out_dir / "deflection_y_cm.png"
        if p_cm.exists():
            saved.append(p_cm)
        print("✔ Zapisano: " + ", ".join(str(s) for s in saved))


if __name__ == "__main__":
    # ---------- kalibracja ----------
    try:
        with np.load(CALIB_PATH) as calib:
            K = calib["camera_matrix"]
            D = calib["dist_coeffs"]
            print(f"✅ Załadowano kalibrację z {CALIB_PATH}")
    except Exception as e:
        print(f"❌ Brak / błąd kalibracji ({e}); kontynuuję bez niej")
        K, D = None, None

    # ---------- katalog wyjściowy serii ----------
    out_dir, series_num = _make_output_dir(SERIES_NO)
    out_video_path = out_dir / OUT_VIDEO
    print(f"📁 Folder wyników: {out_dir} (seria {series_num})")

    ArucoTracker(
        video_path=VIDEO_PATH,
        dict_name=ARUCO_DICT,
        marker_id=MARKER_ID,
        out_path=str(out_video_path),
        camera_matrix=K,
        dist_coeffs=D,
        out_dir=out_dir
    ).run()