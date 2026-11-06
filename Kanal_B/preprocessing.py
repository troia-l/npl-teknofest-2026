"""
Kanal B - Video Ön İşleme Katmanı (preprocessing.py)
===================================================

Bu modül, ham video verilerini multimodal Görsel Dil Modelleri (VLM) tarafından
işlenebilir standart paketlere veya optimize video kliplerine dönüştürür.

İki Farklı Ön İşleme Yolu:
--------------------------
1. **Video Modu Ön İşleme (EVREN VLM İçin Birincil Yol):**
   - TEKNOFEST EVREN VLM modeli, doğrudan video kliplerini kabul eder.
   - :func:`probe_video`: Videonun süre, çözünürlük ve kare hızını decode etmeden inceler.
   - :func:`detect_scene_boundaries`: SSIM yapısal benzerlik farklarına bakarak videoyu
     doğal sahne geçişlerinden keser. Asgari ve azami (60 sn) süre kısıtlarını zorunlu tutar.
   - :func:`extract_video_segment`: Seçilen zaman aralığını GPU (h264_nvenc) veya CPU (libx264)
     donanım hızlandırmasıyla 720p H.264 MP4 formatında dışa aktarır.
   - :func:`prepare_video_segments`: Kısıtlara uyan videoları doğrudan geçirir (prefix cache dostu),
     uzun videoları ise 60 saniyelik kliplere böler.

2. **Grid / Mozaik Ön İşleme (S1b Sözleşmesi — Yerel Modeller İçin Geriye Dönük Yol):**
   - Görsel modeller (vLLM, llama.cpp vb.) için 3 aşamalı akıllı örnekleme uygular:
     - **Aşama 1 (Bulanıklık Filtresi):** Laplacian varyansı düşük (odak dışı/hareket bulanık)
       kareler elenir.
     - **Aşama 2 (Sahne Değişimi Greedy Seçimi):** SSIM farkı yüksek ve aralarında asgari
       zaman boşluğu (`MIN_GAP_SEC`) olan kareler seçilir.
     - **Aşama 3 (Fallback Doldurma):** Hedef kare sayısına (örn. 8) ulaşılamazsa zaman
       ekseninde eşit aralıklı örnekleme yapılır.
     - **Aşama 4 (CLAHE Düşük Işık Düzeltmesi):** Ortalama parlaklığı eşik altında olan
       karanlık karelerin L (parlaklık) kanalına adaptif histogram eşitleme uygulanır.
     - **Aşama 5 (Grid Montajı):** Seçilen kareler 2x4 boyutlarında tek bir mozaik
       JPEG görselinde birleştirilir ve :class:`contracts.VLMFramePacket` (S1b) üretilir.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import av
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import yaml

# Paket içi veya doğrudan çalıştırma uyumlu esnek importlar
try:
    from .video_reader import RawFrame, read_video_stream
    from .contracts import (
        EnhancementInfo,
        FrameMeta,
        FrameQualityMetrics,
        GridLayout,
        VLMFramePacket,
        format_mmss,
    )
except ImportError:
    from video_reader import RawFrame, read_video_stream
    from contracts import (
        EnhancementInfo,
        FrameMeta,
        FrameQualityMetrics,
        GridLayout,
        VLMFramePacket,
        format_mmss,
    )


# ---------------------------------------------------------------------------
# Konfigürasyon Yükleme
# ---------------------------------------------------------------------------

def _load_cfg() -> dict[str, Any]:
    """Proje kökündeki `config.yaml` dosyasının `[preprocessing:]` bloğunu yükler.

    Returns:
        dict[str, Any]: Ön işleme hiperparametrelerini içeren sözlük. Dosya yoksa boş dict.
    """
    cfg_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    )
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        return data.get("preprocessing", {})


_CFG = _load_cfg()

# ---------------------------------------------------------------------------
# Ayarlanabilir Sabitler (Grid ve Örnekleme Parametreleri)
# ---------------------------------------------------------------------------

#: Grid için seçilecek hedef kare sayısı (varsayılan: 8)
TARGET_FRAME_COUNT: int = _CFG.get("target_frame_count", 8)

#: Aday havuzu oluşturulurken kaç karede bir metrik hesaplanacağı (varsayılan: her 5. kare)
CANDIDATE_STRIDE: int = 5

#: Laplacian varyansı bu eşiğin altında kalan kareler bulanık kabul edilir (varsayılan: 40.0)
MIN_SHARPNESS: float = _CFG.get("min_laplacian_variance", 40.0)

#: Seçilen ardışık kareler arasında bulunması gereken asgari zaman farkı (saniye, varsayılan: 1.0)
MIN_GAP_SEC: float = 1.0

#: Ortalama piksel parlaklığı bu değerin altındaysa CLAHE iyileştirmesi tetiklenir (varsayılan: 90.0)
LOW_LIGHT_BRIGHTNESS: float = 90.0

#: CLAHE kontrast sınırlandırma katsayısı (varsayılan: 2.5)
CLAHE_CLIP_LIMIT: float = _CFG.get("clahe_clip_limit", 2.5)

#: CLAHE bölgesel ızgara çözünürlüğü (varsayılan: (8, 8))
CLAHE_TILE_GRID: tuple[int, int] = tuple(_CFG.get("clahe_grid_size", [8, 8]))

#: Grid tekil hücre piksel genişliği ve yüksekliği
_W: int = _CFG.get("frame_width", 384)
_H: int = _CFG.get("frame_height", 384)
CELL_SIZE: tuple[int, int] = (_W, _H)

#: Grid sütun ve satır yerleşimi
GRID_COLS: int = _CFG.get("grid_columns", 4)
GRID_ROWS: int = TARGET_FRAME_COUNT // GRID_COLS


# ---------------------------------------------------------------------------
# İç Veri Yapısı (Yalnızca Grid Aday Seçimi İçin)
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    """Kare seçim algoritmasında değerlendirilen aday kare yapısı.

    Attributes:
        frame (RawFrame): Ham kare referansı (indeks, zaman damgası, piksel matrisi).
        laplacian_var (float): Hesaplanmış keskinlik metriği.
        ssim_diff (float): Bir önceki adaya göre sahne farkı oranı (1.0 - SSIM).
        brightness (float): Karenin ortalama parlaklık değeri (0.0 - 255.0).
    """
    frame: RawFrame
    laplacian_var: float
    ssim_diff: float
    brightness: float


# ---------------------------------------------------------------------------
# Yardımcı Metrik Hesaplama Fonksiyonları
# ---------------------------------------------------------------------------

def _laplacian_var(gray: np.ndarray) -> float:
    """Gri tonlamalı görüntü matrisinden Laplacian varyansı (keskinlik) hesaplar.

    Args:
        gray (np.ndarray): 2 boyutlu uint8 gri tonlamalı görüntü matrisi.

    Returns:
        float: Laplacian filtresinin varyans değeri (yüksek değer = net kenarlar/keskinlik).
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _to_gray_small(rgb: np.ndarray, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    """SSIM hesaplama performansını optimize etmek amacıyla küçültülmüş gri görüntü üretir.

    Args:
        rgb (np.ndarray): (H, W, 3) boyutlu RGB renk uzayındaki piksel dizisi.
        size (tuple[int, int], optional): Hedef (genişlik, yükseklik). Varsayılan: (160, 90).

    Returns:
        np.ndarray: Küçültülmüş gri tonlamalı uint8 matris.
    """
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------------------
# Aşama 1 — Aday Havuzu Oluşturma
# ---------------------------------------------------------------------------

def _collect_candidates(video_path: str) -> list[_Candidate]:
    """Videoyu akış halinde tarayarak her CANDIDATE_STRIDE adımda bir metrikleri hesaplar.

    Tüm videoyu belleğe yüklemeden sadece hesaplanan metrikleri ve aday kare
    referanslarını RAM'de toplar.

    Args:
        video_path (str): Taranacak video dosya yolu.

    Returns:
        list[_Candidate]: Değerlendirilmeye hazır aday kareler listesi.
    """
    candidates: list[_Candidate] = []
    prev_gray = None

    for frame in read_video_stream(video_path):
        if frame.frame_index % CANDIDATE_STRIDE != 0:
            continue

        gray_full = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        gray_small = _to_gray_small(frame.rgb)

        lap = _laplacian_var(gray_full)
        brightness = float(gray_full.mean())

        diff = 1.0 if prev_gray is None else float(1.0 - ssim(prev_gray, gray_small))

        candidates.append(_Candidate(frame, lap, diff, brightness))
        prev_gray = gray_small

    return candidates


# ---------------------------------------------------------------------------
# Aşama 2 ve 3 — Akıllı Kare Seçimi ve Fallback Doldurma
# ---------------------------------------------------------------------------

def _select_frames(
    candidates: list[_Candidate],
) -> tuple[list[_Candidate], set[int]]:
    """Aday havuzundan en bilgilendirici ve keskin kareleri seçer.

    Algoritma Aşamaları:
        1. **Filtre:** Laplacian varyansı < MIN_SHARPNESS olan bulanık kareler elenir.
        2. **Greedy Seçim:** Adaylar SSIM farkına göre büyükten küçüğe sıralanır. Seçilmiş
           karelerle zaman farkı >= MIN_GAP_SEC olanlar havuza eklenir.
        3. **Fallback Doldurma:** Yetersiz kare kaldığında zaman ekseni eşit aralıklarla
           taranarak eksikler tamamlanır.

    Args:
        candidates (list[_Candidate]): İncelenecek aday kare havuzu.

    Returns:
        tuple[list[_Candidate], set[int]]:
            - Seçilen TARGET_FRAME_COUNT adet aday kare listesi (kronolojik sırada).
            - Fallback doldurma ile seçilen adayların Python nesne kimlikleri (`id`) kümesi.
    """
    # Bulanık kareleri filtrele; hepsi bulanıksa filtreyi gevşet
    sharp = [c for c in candidates if c.laplacian_var >= MIN_SHARPNESS] or candidates
    ranked = sorted(sharp, key=lambda c: c.ssim_diff, reverse=True)

    selected: list[_Candidate] = []
    for c in ranked:
        if len(selected) >= TARGET_FRAME_COUNT:
            break
        if all(abs(c.frame.timestamp_sec - s.frame.timestamp_sec) >= MIN_GAP_SEC for s in selected):
            selected.append(c)

    fallback_ids: set[int] = set()

    if len(selected) < TARGET_FRAME_COUNT:
        remaining = TARGET_FRAME_COUNT - len(selected)
        pool = sorted(candidates, key=lambda c: c.frame.timestamp_sec)
        selected_indices = {id(s) for s in selected}
        if pool:
            step = max(1, len(pool) // max(remaining, 1))
            for i in range(0, len(pool), step):
                if len(selected) >= TARGET_FRAME_COUNT:
                    break
                cand = pool[i]
                if id(cand) not in selected_indices:
                    selected.append(cand)
                    selected_indices.add(id(cand))
                    fallback_ids.add(id(cand))

    # Grid montajı için kareleri kronolojik (zaman sırasına) diz
    selected.sort(key=lambda c: c.frame.timestamp_sec)
    return selected[:TARGET_FRAME_COUNT], fallback_ids


# ---------------------------------------------------------------------------
# Aşama 4 — Koşullu CLAHE Kontrast İyileştirmesi
# ---------------------------------------------------------------------------

def _apply_clahe_if_needed(rgb: np.ndarray, brightness: float) -> tuple[np.ndarray, bool]:
    """Yalnızca karanlık karelerde LAB renk uzayının L kanalına CLAHE uygular.

    Aydınlık karelerde aşırı doygunluğu ve gürültüyü önlemek amacıyla işlem yapılmaz.

    Args:
        rgb (np.ndarray): (H, W, 3) RGB renk uzayındaki görüntü matrisi.
        brightness (float): Karenin ortalama parlaklık değeri.

    Returns:
        tuple[np.ndarray, bool]: (İyileştirilmiş RGB matrisi, CLAHE uygulandı mı bayrağı).
    """
    if brightness >= LOW_LIGHT_BRIGHTNESS:
        return rgb, False

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l2 = clahe.apply(l)

    enhanced = cv2.merge((l2, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB), True


# ---------------------------------------------------------------------------
# Aşama 5 — Grid Montajı
# ---------------------------------------------------------------------------

def _build_grid(frames_rgb: list[np.ndarray]) -> np.ndarray:
    """Seçilen kareleri tek bir mozaik (grid) matrisinde birleştirir.

    Eksik kareler bulunması durumunda (çok kısa videolar) boş hücreler siyah
    pikseller ile doldurulur.

    Args:
        frames_rgb (list[np.ndarray]): Grid'e yerleştirilecek kareler listesi.

    Returns:
        np.ndarray: (GRID_ROWS * CELL_SIZE[1], GRID_COLS * CELL_SIZE[0], 3) boyutlu birleşik matris.
    """
    cells = [cv2.resize(f, CELL_SIZE, interpolation=cv2.INTER_AREA) for f in frames_rgb]

    while len(cells) < GRID_ROWS * GRID_COLS:
        cells.append(np.zeros((CELL_SIZE[1], CELL_SIZE[0], 3), dtype=np.uint8))

    rows = [np.hstack(cells[r * GRID_COLS : (r + 1) * GRID_COLS]) for r in range(GRID_ROWS)]
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Grid Modu Ana Giriş Noktası (S1b Paketi Üreticisi)
# ---------------------------------------------------------------------------

def build_vlm_frame_packet(video_path: str, video_id: str, output_dir: str) -> VLMFramePacket:
    """Uçtan uca grid ön işleme akışını yürüterek S1b sözleşme paketini üretir.

    1. Aday kareleri toplar ve metrikleri hesaplar.
    2. Keskinlik ve sahne değişimlerine göre en iyi TARGET_FRAME_COUNT kareyi seçer.
    3. Gerekli karelere CLAHE uygular.
    4. Kareleri 2x4 mozaik JPEG görseline birleştirir ve diske kaydeder.
    5. :class:`contracts.VLMFramePacket` nesnesini oluşturup döndürür.

    Args:
        video_path (str): Kaynak video dosyası.
        video_id (str): Video kimliği.
        output_dir (str): Grid JPEG dosyasının kaydedileceği klasör.

    Returns:
        VLMFramePacket: VLM backend'e sunulmaya hazır S1b paketi.

    Raises:
        ValueError: Videodan hiçbir kare okunamazsa.
    """
    candidates = _collect_candidates(video_path)
    if not candidates:
        raise ValueError("Video karesi okunamadı (boş ya da bozuk dosya olabilir).")

    selected, fallback_ids = _select_frames(candidates)

    enhanced_rgb: list[np.ndarray] = []
    clahe_used_any = False
    frame_metas: list[FrameMeta] = []

    for i, c in enumerate(selected):
        rgb, used_clahe = _apply_clahe_if_needed(c.frame.rgb, c.brightness)
        clahe_used_any = clahe_used_any or used_clahe
        enhanced_rgb.append(rgb)

        if id(c) in fallback_ids:
            reason = "fallback_fill"
        elif c.ssim_diff >= 0.15:
            reason = "scene_change"
        else:
            reason = "uniform_sample"

        frame_metas.append(FrameMeta(
            frame_index=c.frame.frame_index,
            timestamp_sec=c.frame.timestamp_sec,
            grid_position=i,
            selection_reason=reason,
            quality=FrameQualityMetrics(
                laplacian_var=c.laplacian_var,
                ssim_diff=c.ssim_diff,
                brightness_mean=c.brightness,
            ),
        ))

    grid_img = _build_grid(enhanced_rgb)
    os.makedirs(output_dir, exist_ok=True)
    packet_id = str(uuid.uuid4())
    grid_path = os.path.join(output_dir, f"{packet_id}_grid.jpg")

    cv2.imwrite(
        grid_path,
        cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )

    return VLMFramePacket(
        video_id=video_id,
        source_start_sec=selected[0].frame.timestamp_sec,
        source_end_sec=selected[-1].frame.timestamp_sec,
        frames=frame_metas,
        grid_layout=GridLayout(rows=GRID_ROWS, cols=GRID_COLS, cell_size=CELL_SIZE),
        enhancement=EnhancementInfo(
            clahe_applied=clahe_used_any,
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_grid_size=CLAHE_TILE_GRID,
        ),
        grid_image_path=grid_path,
        packet_id=packet_id,
    )


# ===========================================================================
# VİDEO MODU ÖN İŞLEME — Sahne Kesimi ve 720p/60sn Segment Üretimi
# ===========================================================================

#: Sahne değişimi eşiği: (1.0 - SSIM) bu değeri aşarsa kesim adayı kabul edilir
SCENE_SSIM_THRESHOLD: float = _CFG.get("scene_ssim_threshold", 0.30)

#: Bir segmentin sahip olabileceği asgari süre (saniye)
SCENE_MIN_SEGMENT_SEC: float = _CFG.get("scene_min_segment_sec", 15.0)

#: Segment üst sınırı — EVREN video kısıtı (720p için azami 60 saniye)
SCENE_MAX_SEGMENT_SEC: float = _CFG.get("scene_max_segment_sec", 60.0)

#: Sahne değişim taramasında kaç karede bir SSIM hesaplanacağı
SCENE_DETECT_STRIDE: int = _CFG.get("scene_detect_stride", 5)

#: Segment klipleri için hedef azami dikey piksel çözünürlüğü (720p kısıtı)
VLM_MAX_HEIGHT: int = _CFG.get("vlm_max_height", 720)

#: H.264 video kodlama CRF (kalite) ve hız ön ayarları
VLM_CLIP_CRF: int = _CFG.get("vlm_clip_crf", 23)
VLM_CLIP_PRESET: str = _CFG.get("vlm_clip_preset", "veryfast")


@dataclass
class VideoClip:
    """VLM servisine doğrudan gönderilmeye hazır tek bir video parçasını temsil eder.

    Attributes:
        path (str): Klip MP4 dosyasının mutlak dosya yolu.
        start_sec (float): Orijinal tam videodaki mutlak başlangıç saniyesi.
        end_sec (float): Orijinal tam videodaki mutlak bitiş saniyesi.
        index (int): Segmentin kronolojik sıra indeksi (0-tabanlı).
        reencoded (bool): Dosyanın geçici olarak yeniden kodlanıp kodlanmadığı.
            False ise orijinal dosyanın kendisidir ve silinmemelidir.
    """
    path: str
    start_sec: float
    end_sec: float
    index: int = 0
    reencoded: bool = False

    @property
    def duration_sec(self) -> float:
        """Klip süresini saniye cinsinden döndürür."""
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def time_label(self) -> str:
        """Klibin 'MM:SS-MM:SS' formatındaki zaman aralığı etiketi."""
        return f"{format_mmss(self.start_sec)}-{format_mmss(self.end_sec)}"


def probe_video(video_path: str) -> dict[str, Any]:
    """Video dosyasını decode etmeden hızlıca konteyner meta verilerini okur.

    Args:
        video_path (str): İncelenecek video dosyası yolu.

    Returns:
        dict[str, Any]: `width`, `height`, `fps`, `duration_sec` değerlerini içeren sözlük.

    Raises:
        FileNotFoundError: Video dosyası mevcut değilse.
        av.AVError: Video akışı okunamıyorsa.
    """
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0

        duration = 0.0
        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = float(container.duration) / av.time_base
        if not duration and stream.frames and fps:
            duration = stream.frames / fps

        return {
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "fps": fps,
            "duration_sec": float(duration),
        }


def detect_scene_boundaries(
    video_path: str,
    ssim_threshold: float | None = None,
    min_segment_sec: float | None = None,
    max_segment_sec: float | None = None,
    stride: int | None = None,
) -> list[tuple[float, float]]:
    """Videoyu görsel sahne geçişlerinden faydalanarak zaman aralıklarına böler.

    Kesim anları rastgele zamanlar yerine görsel kompozisyonun değiştiği anlara
    denk getirilir; böylece bir olayın ortadan ikiye bölünmesi engellenir.
    Ancak `max_segment_sec` sert bir tavan olarak korunur.

    Args:
        video_path (str): Bölünecek video dosya yolu.
        ssim_threshold (float | None, optional): 1-SSIM sahne değişim eşiği. None ise config kullanılır.
        min_segment_sec (float | None, optional): Asgari segment süresi (sn). None ise config kullanılır.
        max_segment_sec (float | None, optional): Azami segment süresi (sn). None ise config kullanılır.
        stride (int | None, optional): Kaç karede bir SSIM hesaplanacağı. None ise config kullanılır.

    Returns:
        list[tuple[float, float]]: Sıralı `(start_sec, end_sec)` zaman aralıkları listesi.

    Raises:
        ValueError: Video süresi tespit edilemezse.
    """
    ssim_threshold = SCENE_SSIM_THRESHOLD if ssim_threshold is None else ssim_threshold
    min_segment_sec = SCENE_MIN_SEGMENT_SEC if min_segment_sec is None else min_segment_sec
    max_segment_sec = SCENE_MAX_SEGMENT_SEC if max_segment_sec is None else max_segment_sec
    stride = SCENE_DETECT_STRIDE if stride is None else stride

    duration = probe_video(video_path)["duration_sec"]
    if duration <= 0:
        raise ValueError(f"Video süresi belirlenemedi: {video_path}")

    # Video zaten azami süre sınırının altındaysa bölme işlemi yapılmaz
    if duration <= max_segment_sec:
        return [(0.0, duration)]

    # Sahne değişimi zaman damgalarını topla
    changes: list[float] = []
    prev_gray = None
    for frame in read_video_stream(video_path):
        if frame.frame_index % stride != 0:
            continue
        gray = _to_gray_small(frame.rgb)
        if prev_gray is not None:
            if float(1.0 - ssim(prev_gray, gray)) >= ssim_threshold:
                changes.append(frame.timestamp_sec)
        prev_gray = gray

    # Zaman çizelgesini yürüyerek segment sınırlarını belirle
    segments: list[tuple[float, float]] = []
    start = 0.0
    cursor = 0
    while start < duration - 1e-6:
        hard_end = min(start + max_segment_sec, duration)

        while cursor < len(changes) and changes[cursor] <= start + min_segment_sec:
            cursor += 1

        if cursor < len(changes) and changes[cursor] < hard_end:
            end = changes[cursor]
            cursor += 1
        else:
            end = hard_end

        segments.append((start, end))
        start = end

    # Çok kısa kalan son segmenti bir öncekine dahil et (üst sınır aşılmıyorsa)
    if len(segments) >= 2:
        last_start, last_end = segments[-1]
        if (last_end - last_start) < min_segment_sec:
            prev_start, _ = segments[-2]
            if (last_end - prev_start) <= max_segment_sec:
                segments[-2] = (prev_start, last_end)
                segments.pop()

    return segments


def extract_video_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    out_path: str,
    max_height: int | None = None,
) -> str:
    """Videonun belirli bir aralığını keser, 720p'ye ölçeklendirir ve yeni MP4 olarak kaydeder.

    Donanım Hızlandırma Desteği:
        Sistemde NVIDIA GPU ve FFmpeg NVENC desteği (`h264_nvenc`) mevcutsa donanım
        hızlandırma ile çok hızlı kodlama yapılır; bulunamazsa standart CPU `libx264`
        kodlayıcısına güvenli fallback yapılır.

    Args:
        video_path (str): Kaynak video dosyası yolu.
        start_sec (float): Kesimin başlangıç saniyesi.
        end_sec (float): Kesimin bitiş saniyesi.
        out_path (str): Üretilecek MP4 klibinin kayıt yolu.
        max_height (int | None, optional): Hedef azami dikey çözünürlük (varsayılan: 720).

    Returns:
        str: Kaydedilen klibin dosya yolu (`out_path`).

    Raises:
        ValueError: Belirtilen zaman aralığında hiç kare bulunamazsa.
    """
    max_height = VLM_MAX_HEIGHT if max_height is None else max_height
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with av.open(video_path) as inp:
        istream = inp.streams.video[0]
        istream.thread_type = "AUTO"

        in_w = int(istream.codec_context.width)
        in_h = int(istream.codec_context.height)
        rate = istream.average_rate or Fraction(25, 1)
        time_base = float(istream.time_base) if istream.time_base else 1.0

        scale = min(1.0, max_height / in_h) if in_h else 1.0
        out_w = max(2, (int(in_w * scale) // 2) * 2)
        out_h = max(2, (int(in_h * scale) // 2) * 2)

        if start_sec > 0 and time_base:
            try:
                inp.seek(int(start_sec / time_base), stream=istream)
            except av.AVError:
                inp.seek(0)

        codec_name = "libx264"
        codec_opts = {"crf": str(VLM_CLIP_CRF), "preset": VLM_CLIP_PRESET}
        try:
            if "h264_nvenc" in av.codecs_available:
                codec_name = "h264_nvenc"
                codec_opts = {
                    "preset": "p4",
                    "tune": "hq",
                    "rc": "vbr",
                    "cq": str(VLM_CLIP_CRF),
                    "b": "2.5M",
                    "maxrate": "5M",
                }
        except Exception:
            pass

        out = av.open(out_path, mode="w")
        try:
            try:
                ostream = out.add_stream(codec_name, rate=rate)
                ostream.options = codec_opts
            except Exception:
                ostream = out.add_stream("libx264", rate=rate)
                ostream.options = {"crf": str(VLM_CLIP_CRF), "preset": VLM_CLIP_PRESET}

            ostream.width = out_w
            ostream.height = out_h
            ostream.pix_fmt = "yuv420p"
            ostream.time_base = Fraction(1, 90000)

            written = 0
            for frame in inp.decode(istream):
                if frame.pts is None:
                    continue
                t = float(frame.pts) * time_base
                if t < start_sec - 1e-3:
                    continue
                if t >= end_sec:
                    break

                new_frame = frame.reformat(width=out_w, height=out_h, format="yuv420p")
                new_frame.pts = int(round((t - start_sec) / float(ostream.time_base)))
                new_frame.time_base = ostream.time_base

                for packet in ostream.encode(new_frame):
                    out.mux(packet)
                written += 1

            for packet in ostream.encode():
                out.mux(packet)
        finally:
            out.close()

    if written == 0:
        raise ValueError(
            f"{start_sec:.1f}-{end_sec:.1f} sn aralığında kare bulunamadı: {video_path}"
        )
    return out_path


def prepare_video_segments(
    video_path: str,
    output_dir: str,
    max_segment_sec: float | None = None,
    max_height: int | None = None,
    ssim_threshold: float | None = None,
    min_segment_sec: float | None = None,
) -> list[VideoClip]:
    """Videoyu EVREN `vlm` modeline gönderilmeye hazır :class:`VideoClip` parçalarına dönüştürür.

    Optimizasyon Mantığı:
        - Video süre (<= 60 sn) ve çözünürlük (<= 720p) kriterlerine zaten uyuyorsa,
          dosya kesinlikle yeniden kodlanmaz (`reencoded=False`). Bu sayede hem zaman
          kaybı önlenir hem de aynı videoya yapılacak sorgularda VLM sunucusunun
          ön ek önbelleği (prefix caching) tam isabetle çalışır.
        - Kriterleri aşan videolar ise sahne sınırlarından bölünerek her parça 720p
          H.264 klibi olarak dışa aktarılır (`reencoded=True`).

    Args:
        video_path (str): Kaynak video dosyası.
        output_dir (str): Segment MP4 kliplerinin kaydedileceği dizin.
        max_segment_sec (float | None, optional): Azami segment süresi. None ise config: 60.0.
        max_height (int | None, optional): Azami dikey çözünürlük. None ise config: 720.
        ssim_threshold (float | None, optional): Sahne kesim SSIM eşiği.
        min_segment_sec (float | None, optional): Asgari segment süresi.

    Returns:
        list[VideoClip]: Zaman sırasına göre sıralanmış klip listesi.

    Raises:
        ValueError: Video süresi hesaplanamazsa.
    """
    max_segment_sec = SCENE_MAX_SEGMENT_SEC if max_segment_sec is None else max_segment_sec
    max_height = VLM_MAX_HEIGHT if max_height is None else max_height

    info = probe_video(video_path)
    duration = info["duration_sec"]
    if duration <= 0:
        raise ValueError(f"Video süresi belirlenemedi: {video_path}")

    fits_duration = duration <= max_segment_sec
    fits_resolution = info["height"] <= max_height
    if fits_duration and fits_resolution:
        return [VideoClip(
            path=video_path, start_sec=0.0, end_sec=duration, index=0, reencoded=False
        )]

    ranges = detect_scene_boundaries(
        video_path,
        ssim_threshold=ssim_threshold,
        min_segment_sec=min_segment_sec,
        max_segment_sec=max_segment_sec,
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]

    clips: list[VideoClip] = []
    for idx, (start, end) in enumerate(ranges):
        out_path = os.path.join(output_dir, f"{stem}_seg{idx:03d}.mp4")
        extract_video_segment(video_path, start, end, out_path, max_height=max_height)
        clips.append(VideoClip(
            path=out_path, start_sec=start, end_sec=end, index=idx, reencoded=True
        ))
    return clips
