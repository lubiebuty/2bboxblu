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
import sys, time
from typing import List, Tuple, Dict

# ------------ Ustawienia -------------
VIDEO_PATH = "/Users/bartlomiejostasz/lot/n/git/1.mov"
# plik kalibracyjny kamery (macierz K i dystorsja)
CALIB_PATH = '/Users/bartlomiejostasz/PYCH/LOT/1:5.npz'
ARUCO_DICT = "DICT_6X6_1000"   # ręcznie lub AUTO w przyszłości
MARKER_ID  = 2                 # None = pierwszy wykryty
OUT_VIDEO  = "out_annotated.mp4"
MARKER_LEN_M = 0.04            # wymiar boku markera w metrach (tu: 4 cm)
# -------------------------------------

# słowniki OpenCV
ARUCO_DICTS: Dict[str, int] = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco) if name.startswith("DICT_")
}


class ArucoTracker:
    def __init__(
                 self,
                 video_path: str | int,
                 dict_name: str,
                 marker_id: int | None,
                 out_path: str,
                 camera_matrix=None,
                 dist_coeffs=None) -> None:
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

        self.records: List[Tuple[float,float,float,float | None]] = []
        self.first_center: Tuple[float,float] | None = None

        self.K = camera_matrix
        self.D = dist_coeffs

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

        # --- Lucas–Kanade (LK) fallback i ROI re-detect ---
        self.lk_active = False
        self.prev_gray = None
        self.prev_pts = None
        self.lk_winSize = (15, 15)
        self.lk_maxLevel = 3
        self.lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        self.roi_expand = 1.6  # przy re-detect powiększamy ostatni bbox

        # --- BLUE ROI (łańcuch) + bezpieczeństwo ---
        self.blue_roi = None                    # (x,y,w,h)
        self.blue_roi_half = 40                 # stałe ±40 px od środka (podstawowe okno)
        self.blue_roi_half_big = 80             # „winda bezpieczeństwa” po 1 missie
        self.blue_miss = 0                      # kolejne missy w BLUE ROI
        self.anchor_every = 10                  # co N klatek próba full-frame kotwicy

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
        corners, ids, _ = self._detect(roi)
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

    def _detect_in_bbox(self, frame: np.ndarray, bbox: Tuple[int,int,int,int], id_wanted: int | None):
        """Wykryj ArUco w *dokładnym* bboxie (bez dodatkowego skalowania)."""
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None, None
        corners, ids, _ = self._detect(roi)
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
                    c_big, _ = self._detect_in_bbox(gray, big_box, self.marker_id)
                    if c_big is not None:
                        c = c_big; detected = True; used_blue = True; self.blue_miss = 0
                    else:
                        self.blue_miss += 1
                        # 1b) Po 2 missach – kotwica pełnokadrowa
                        if self.blue_miss >= 2:
                            corners, ids, _ = self._detect(gray)
                            if ids is not None and len(ids):
                                ids = ids.flatten(); idx = 0
                                if self.marker_id is not None:
                                    matches = np.where(ids == self.marker_id)[0]
                                    if len(matches): idx = int(matches[0])
                                    else: ids = None
                                if ids is not None and len(ids):
                                    c = corners[idx][0]; detected = True; used_blue = False; self.blue_miss = 0
            else:
                # 0) Pierwsza kotwica: pełny kadr
                corners, ids, _ = self._detect(gray)
                if ids is not None and len(ids):
                    ids = ids.flatten(); idx = 0
                    if self.marker_id is not None:
                        matches = np.where(ids == self.marker_id)[0]
                        if len(matches): idx = int(matches[0])
                        else: ids = None
                    if ids is not None and len(ids):
                        c = corners[idx][0]; detected = True

            # 0b) Dodatkowa kotwica co N klatek (gdy korzystaliśmy z BLUE ROI)
            if detected and used_blue and self.anchor_every and (frame_idx % self.anchor_every == 0):
                corners_a, ids_a, _ = self._detect(gray)
                if ids_a is not None and len(ids_a):
                    ids_a = ids_a.flatten(); idx_a = 0
                    if self.marker_id is not None:
                        matches = np.where(ids_a == self.marker_id)[0]
                        if len(matches): idx_a = int(matches[0])
                        else: ids_a = None
                    if ids_a is not None and len(ids_a):
                        c = corners_a[idx_a][0]

            if detected and c is not None:
                self.lost_counter = 0

                # Środek markera: z geometrii jeśli mamy kalibrację, inaczej centroid narożników
                if self.use_undistort:
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers([c], MARKER_LEN_M, self.newK, self.zeroD)
                    center_img, _ = cv2.projectPoints(np.array([[0., 0., 0.]], np.float32), rvec[0], tvec[0], self.newK, self.zeroD)
                    cx, cy = map(float, center_img.ravel())
                    z_cm = float(tvec[0][0][2] * 100.0)
                else:
                    cx, cy = c.mean(axis=0)

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
                cv2.polylines(frame_vis, [c.astype(int)], True, (0, 255, 0), 2)
                cv2.circle(frame_vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                self.records.append((t, cx, cy, z_cm))

                # --- AKTUALIZUJ BLUE ROI (zawsze) ---
                half = self.blue_roi_half
                bx = int(round(cx - half)); by = int(round(cy - half))
                bw = int(2*half); bh = int(2*half)
                self.blue_roi = self._clip_bbox(bx, by, bw, bh, self.w, self.h)
                # rysuj niebieski box
                x, y, w, h = self.blue_roi
                cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
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
                        # Zachowujemy pełny porządek 4 narożników (tak jak w prev_pts)
                        valid = st.flatten() == 1
                        n_valid = int(valid.sum())
                        if n_valid >= 3:
                            used_lk = True
                            curr_pts = self.prev_pts.copy()  # 4x1x2
                            curr_pts[valid] = next_pts[valid]

                            if n_valid == 3:
                                # Odtwórz brakujący narożnik z transformacji afinicznej (3 korespondencje)
                                src = self.prev_pts[valid].reshape(3,2).astype(np.float32)
                                dst = next_pts[valid].reshape(3,2).astype(np.float32)
                                M = cv2.getAffineTransform(src, dst)  # 2x3
                                miss_idx = int(np.where(~valid)[0][0])
                                p = self.prev_pts[miss_idx,0].astype(np.float32)  # (x,y) poprzedni
                                pred = (M @ np.array([p[0], p[1], 1.0], dtype=np.float32)).reshape(2)
                                curr_pts[miss_idx,0] = pred

                            c_lk = curr_pts.reshape(-1, 2)
                            cx = float(np.mean(c_lk[:, 0]))
                            cy = float(np.mean(c_lk[:, 1]))

                            x_min, y_min = np.min(c_lk, axis=0)
                            x_max, y_max = np.max(c_lk, axis=0)
                            pad = 15
                            x_min = max(0, int(x_min) - pad)
                            y_min = max(0, int(y_min) - pad)
                            x_max = min(self.w - 1, int(x_max) + pad)
                            y_max = min(self.h - 1, int(y_max) + pad)
                            bbox = (
                                x_min, y_min, max(2, x_max - x_min), max(2, y_max - y_min)
                            )
                            self.last_bbox = bbox

                            # update stanu LK (pełny zestaw 4 punktów)
                            self.prev_gray = gray.copy()
                            self.prev_pts = curr_pts

                            # rysowanie zawsze 4 punktów
                            cv2.polylines(frame_vis, [c_lk.astype(int)], True, (0, 255, 255), 2)
                            cv2.circle(frame_vis, (int(cx), int(cy)), 4, (0, 255, 255), -1)
                            self.records.append((t, cx, cy, None))
                            # Aktualizuj BLUE ROI na podstawie LK (co klatkę)
                            half = self.blue_roi_half
                            bx = int(round(cx - half)); by = int(round(cy - half))
                            bw = int(2*half); bh = int(2*half)
                            self.blue_roi = self._clip_bbox(bx, by, bw, bh, self.w, self.h)
                            cv2.rectangle(frame_vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
                            blue_updated = True
                            state = "LK"
                        else:
                            # Za mało punktów → wyłącz LK i przejdź do CSRT
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
                            self.records.append((t, cx, cy, None))
                            # Aktualizuj BLUE ROI z CSRT (co klatkę)
                            half = self.blue_roi_half
                            bx = int(round(cx - half)); by = int(round(cy - half))
                            bw = int(2*half); bh = int(2*half)
                            self.blue_roi = self._clip_bbox(bx, by, bw, bh, self.w, self.h)
                            cv2.rectangle(frame_vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
                            blue_updated = True
                        else:
                            cv2.putText(frame_vis, "LOST", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame_vis, "LOST", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Jeśli w tej klatce nie udało się zaktualizować BLUE ROI (brak DETECT/LK/CSRT),
            # przesuń go ekstrapolacją z dwóch ostatnich zapisów (stała prędkość).
            if not blue_updated:
                if len(self.records) >= 2:
                    cx_prev, cy_prev = self.records[-2][1], self.records[-2][2]
                    cx_last, cy_last = self.records[-1][1], self.records[-1][2]
                    cx_pred = 2*cx_last - cx_prev
                    cy_pred = 2*cy_last - cy_prev
                    half = self.blue_roi_half
                    bx = int(round(cx_pred - half)); by = int(round(cy_pred - half))
                    bw = int(2*half); bh = int(2*half)
                    self.blue_roi = self._clip_bbox(bx, by, bw, bh, self.w, self.h)
                    # dorysuj dla podglądu w MP4
                    cv2.rectangle(frame_vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
                    blue_updated = True

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
        # przygotuj ramkę danych
        df = pd.DataFrame(self.records, columns=["t","cx","cy","z_cm"])
        if self.first_center is not None:
            df["dy_px"] = self.first_center[1] - df["cy"]
        else:
            df["dy_px"] = 0.0
        # zapisz CSV
        df.to_csv("positions_px.csv", index=False)
        # wykres wychylenia pionowego
        plt.figure(figsize=(8,4))
        plt.plot(df["t"], df["dy_px"])
        plt.axhline(0, linewidth=0.8)
        plt.xlabel("Czas [s]"); plt.ylabel("Δy [px]")
        plt.title("Wychylenie pionowe markera")
        plt.grid(True); plt.tight_layout()
        plt.savefig("deflection_px.png", dpi=150)
        plt.close()
        # wykres Z, jeśli dostępny
        if df["z_cm"].notna().any():
            plt.figure(figsize=(8,4))
            plt.plot(df["t"], df["z_cm"])
            plt.xlabel("Czas [s]"); plt.ylabel("Z [cm]")
            plt.title("Przemieszczenie osi Z")
            plt.grid(True); plt.tight_layout()
            plt.savefig("deflection_z_cm.png", dpi=150)
            plt.close()
        print("✔ Zapisano: positions_px.csv, deflection_px.png" + (" i deflection_z_cm.png" if df["z_cm"].notna().any() else ""))


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

    ArucoTracker(
        video_path=VIDEO_PATH,
        dict_name=ARUCO_DICT,
        marker_id=MARKER_ID,
        out_path=OUT_VIDEO,
        camera_matrix=K,
        dist_coeffs=D
    ).run()
