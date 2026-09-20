from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen
import cv2
import numpy as np
from PIL import Image

CASCADE_URL = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASCADE_PATH = PROJECT_ROOT / "models" / "face_detector" / "lbpcascade_animeface.xml"

@dataclass
class FaceRegion:
    x: int; y: int; w: int; h: int; confidence: float = 1.0; support: int = 1
    @property
    def cx(self): return self.x + self.w / 2
    @property
    def cy(self): return self.y + self.h / 2

@dataclass
class PersonRegion:
    x: int; y: int; w: int; h: int; confidence: float; mask: Optional[np.ndarray] = None
    @property
    def cx(self): return self.x + self.w / 2
    @property
    def cy(self): return self.y + self.h / 2

class FaceDetector:
    def __init__(self, cascade_path: Path = CASCADE_PATH):
        self.cascade_path = Path(cascade_path); self.cascade = None
        self._ensure_cascade()

    def _ensure_cascade(self):
        self.cascade_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cascade_path.exists() or self.cascade_path.stat().st_size < 50_000:
            print("[FaceDetector] Anime face model missing/incomplete; downloading...", flush=True)
            tmp = self.cascade_path.with_suffix('.xml.part')
            try:
                with urlopen(CASCADE_URL, timeout=30) as r, open(tmp,'wb') as out:
                    while True:
                        chunk=r.read(64*1024)
                        if not chunk: break
                        out.write(chunk)
                if tmp.stat().st_size < 50_000: raise RuntimeError('downloaded cascade is incomplete')
                tmp.replace(self.cascade_path)
            except Exception as exc:
                print(f"[FaceDetector] Could not download anime face model: {exc}", flush=True); return
        try:
            self.cascade = cv2.CascadeClassifier(str(self.cascade_path))
            if self.cascade.empty():
                self.cascade = None; print(f"[FaceDetector] Invalid cascade: {self.cascade_path}", flush=True)
            else: print(f"[FaceDetector] Anime face detector ready: {self.cascade_path}", flush=True)
        except Exception as exc:
            print(f"[FaceDetector] OpenCV face detector failed: {exc}", flush=True)

    def detect(self, image: Image.Image, expected_faces: Optional[int] = None) -> List[FaceRegion]:
        """Detect readable anime faces and collapse duplicate/weak cascade hits.

        The LBP anime cascade is deliberately run at several scales/contrast variants,
        but those runs often produce several boxes for the same face.  We first cluster
        overlapping/nearby hits and score each cluster by repeated detector support and
        face size.  ``expected_faces`` is a soft scene-level hint (from JoyTag gender
        counts), not a command to invent faces.
        """
        if self.cascade is None:
            return []
        rgb = np.asarray(image.convert('RGB'))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        min_dim = min(w, h)
        # A 1.2% minimum is too permissive for anime illustrations and is the main
        # source of tiny false positives.  Keep it adaptive so small characters can
        # still be found in large images.
        min_face = max(22, int(min_dim * 0.0175))
        raw = []
        variants = [gray, cv2.equalizeHist(gray)]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        variants.append(clahe.apply(gray))

        for source in variants:
            for scale in (1.0, 1.5, 2.0):
                work = (cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                        if scale != 1.0 else source)
                mf = max(16, int(min_face * scale))
                for sf, neighbors in ((1.05, 4), (1.08, 3), (1.12, 2)):
                    found = self.cascade.detectMultiScale(
                        work, scaleFactor=sf, minNeighbors=neighbors, minSize=(mf, mf)
                    )
                    for x, y, fw, fh in found:
                        raw.append(FaceRegion(
                            round(x / scale), round(y / scale),
                            round(fw / scale), round(fh / scale),
                            1.0, 1
                        ))

        # Merge repeated detections of the same face.  A small centre-distance allowance
        # handles boxes from different cascade scales that do not overlap enough for IoU.
        clusters: list[list[FaceRegion]] = []
        for candidate in raw:
            if candidate.w < min_face or candidate.h < min_face:
                continue
            ratio = candidate.w / max(candidate.h, 1)
            if not 0.55 <= ratio <= 1.8:
                continue
            placed = False
            for cluster in clusters:
                representative = max(cluster, key=lambda r: r.w * r.h)
                center_dist = ((candidate.cx - representative.cx) ** 2 +
                               (candidate.cy - representative.cy) ** 2) ** 0.5
                scale_ref = max(representative.w, representative.h, candidate.w, candidate.h)
                if self._iou(candidate, representative) >= 0.18 or center_dist <= scale_ref * 0.42:
                    cluster.append(candidate)
                    placed = True
                    break
            if not placed:
                clusters.append([candidate])

        candidates = []
        for cluster in clusters:
            # Prefer the median-sized box; extreme boxes are commonly cascade artefacts.
            cluster_sorted = sorted(cluster, key=lambda r: r.w * r.h)
            representative = cluster_sorted[len(cluster_sorted) // 2]
            support = len(cluster)
            area_ratio = (representative.w * representative.h) / max(w * h, 1)
            # Repeated detections are much stronger evidence than a single tiny hit.
            score = support * 2.0 + min(area_ratio * 140.0, 8.0)
            candidates.append(FaceRegion(
                representative.x, representative.y, representative.w, representative.h,
                score, support
            ))

        candidates.sort(key=lambda r: (r.confidence, r.w * r.h), reverse=True)
        # Final spatial NMS.
        regions = []
        for c in candidates:
            if any(self._iou(c, k) > 0.30 for k in regions):
                continue
            regions.append(c)

        # Remove isolated, tiny detections when stronger faces exist.  We do not require
        # a particular count: a hidden/body-only character should produce no row.
        if regions:
            largest = max(r.w * r.h for r in regions)
            filtered = [
                r for r in regions
                if r.support >= 2 or r.w * r.h >= largest * 0.10
            ]
            regions = filtered or regions[:1]

        # Scene counts are useful as a soft upper bound.  Never create synthetic faces.
        if expected_faces and expected_faces > 0 and len(regions) > expected_faces:
            selected = []
            # Greedily maximise detector score while enforcing a reasonable separation.
            for r in regions:
                if len(selected) >= expected_faces:
                    break
                if not selected or all(self._center_distance(r, s) > 0.42 * max(r.w, r.h, s.w, s.h) for s in selected):
                    selected.append(r)
            if len(selected) < expected_faces:
                for r in regions:
                    if r not in selected:
                        selected.append(r)
                    if len(selected) >= expected_faces:
                        break
            regions = selected

        # Stable visual order.
        regions.sort(key=lambda r: (r.cx, r.cy))
        if regions:
            details = ", ".join(
                f"({r.x},{r.y},{r.w}x{r.h}; support={r.support})" for r in regions
            )
            print(f"[CharacterDetector] Face candidates: {details}", flush=True)
        out = []
        for r in regions:
            x = max(0, min(w - 1, r.x)); y = max(0, min(h - 1, r.y))
            x2 = max(x + 1, min(w, r.x + r.w)); y2 = max(y + 1, min(h, r.y + r.h))
            rw, rh = x2 - x, y2 - y
            if rw >= min_face and rh >= min_face:
                out.append(FaceRegion(x, y, rw, rh, r.confidence, r.support))
        return out

    @staticmethod
    def _center_distance(a: FaceRegion, b: FaceRegion) -> float:
        return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5

    def character_views(self, image: Image.Image, face: FaceRegion, person=None, all_faces=None) -> dict[str, Image.Image]:
        """Return a small, fixed set of crops for one readable face."""
        full = self._simple_face_box(image, face, all_faces or [face])
        W, H = image.size
        left = max(0, int(face.x - face.w * 1.5))
        top = max(0, int(face.y - face.h * 1.5))
        right = min(W, int(face.x + face.w * 2.5))
        bottom = min(H, int(face.y + face.h * 2.8))
        head = image.crop((left, top, right, bottom))
        return {"head": head, "full": full}

    def crop_person(self, image: Image.Image, face: FaceRegion, person=None, all_faces=None) -> Image.Image:
        return self.character_views(image, face, person, all_faces)["full"]

    @staticmethod
    def _simple_face_box(image: Image.Image, face: FaceRegion, faces: List[FaceRegion]) -> Image.Image:
        W, H = image.size
        left=max(0,int(face.cx-face.w*3.5)); right=min(W,int(face.cx+face.w*3.5))
        top=max(0,int(face.cy-face.h*1.8)); bottom=min(H,int(face.cy+face.h*9.5))
        return image.crop((left,top,right,bottom))

    @staticmethod
    def _nms(candidates:List[FaceRegion],iou_threshold=.28):
        kept=[]
        for c in sorted(candidates,key=lambda r:r.w*r.h,reverse=True):
            if not any(FaceDetector._iou(c,k)>iou_threshold for k in kept):kept.append(c)
        return kept
    @staticmethod
    def _iou(a,b):
        ax2,ay2=a.x+a.w,a.y+a.h; bx2,by2=b.x+b.w,b.y+b.h; ix1,iy1=max(a.x,b.x),max(a.y,b.y); ix2,iy2=min(ax2,bx2),min(ay2,by2); iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
        return inter/(a.w*a.h+b.w*b.h-inter) if inter else 0.0
