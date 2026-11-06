"""
Kanal B - Video Okuyucu Modülü (video_reader.py)
================================================

Bu modül, video dosyalarından ham görüntü karelerini (raw frames) ve zaman
damgası meta verilerini yüksek doğrulukla çözümlemek (decode etmek) için PyAV
(FFmpeg C-API Python bağlamları) kütüphanesini kullanır.

Mimari Rolü ve Veri Akışı:
--------------------------
Kanal A ve Kanal B ortak video akış kaynağı olarak bu jeneratörü paylaşır:
    [Video Dosyası]
           │
           ▼
    read_video_stream() / VideoReader
           │
           ▼
        RawFrame (RGB24 numpy dizisi, PTS tabanlı saniye damgası, sıra no)
           │
           ├──► Kanal A (Nesne Tespiti, Takip, Hareket Analizi)
           └──► Kanal B (preprocessing.py: SSIM analizi, Keskinlik filtreleme, Kırpma)

Neden OpenCV (cv2.VideoCapture) Yerine PyAV (FFmpeg)?
-----------------------------------------------------
1. **Değişken Kare Hızı (VFR - Variable Frame Rate) Uyumluluğu:**
   - Standart OpenCV `cv2.VideoCapture.get(CAP_PROP_POS_MSEC)` değişken kare hızlı
     veya bazı MP4/MKV konteynerlerinde zaman damgalarını (PTS) yanlış hesaplayabilir
     ya da kare atlayabilir.
   - PyAV, doğrudan FFmpeg dekoder seviyesindeki PTS (Presentation Time Stamp) ve
     `time_base` oranına erişir. Bu sayede her karenin video başlangıcına göre
     gerçek mutlak saniyesi (`timestamp_sec`) tam doğrulukla hesaplanır.

2. **Çok Çekirdekli Decode (Multithreading):**
   - FFmpeg dekoderi `thread_type="AUTO"` ile çok çekirdekli donanım/CPU optimizasyonu
     sağlar.

3. **Bellek Dostu Akış (Streaming):**
   - Tüm videoyu RAM'e yüklemek yerine jeneratör modeliyle kare kare akış sunarak
     bellek tüketimini minimumda tutar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# RawFrame — Kanallar Arası Ortak Kare Veri Yapısı
# ---------------------------------------------------------------------------

@dataclass
class RawFrame:
    """Tek bir video karesine ait piksel verisi ve zamanlama meta bilgileri.

    Bu veri yapısı, videodan okunan ham kareden türetilir ve Kanal B ön işleme
    katmanında (:class:`preprocessing._Candidate`) paketlenerek kullanılır.
    Aynı zamanda Kanal A'nın örnekleme döngüsünde de paylaşılan ortak veri sözleşmesidir.

    Attributes:
        frame_index (int): Videonun 0-indeksli ardışık kare numarası. Adım (stride)
            filtrelemeleri için temel referanstır.
        timestamp_sec (float): FFmpeg PTS değerinden saniyeye çevrilen gerçek zaman
            konumu. Kareler arası asgari süre (MIN_GAP_SEC) kontrolleri bu alana dayanır.
        rgb (NDArray[np.uint8]): (H, W, 3) boyutlarında `uint8` tipinde RGB renk
            uzayındaki numpy dizisi.
            .. note::
                OpenCV fonksiyonları varsayılan olarak BGR formatı bekler. Görüntüyü
                OpenCV'ye aktarırken `cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)`
                dönüşümü gerekebilir.
    """
    frame_index: int
    timestamp_sec: float
    rgb: NDArray[np.uint8]


# ---------------------------------------------------------------------------
# VideoReader — Nesne Yönelimli Video Okuyucu Sınıfı
# ---------------------------------------------------------------------------

class VideoReader:
    """PyAV tabanlı güvenli, bağlam yöneticisi (context manager) destekli video okuyucu.

    Gelişmiş video okuma, meta veri sorgulama (toplam kare, FPS, süre) ve özel
    aralık iterasyonları için tasarlanmıştır.

    Examples:
        >>> with VideoReader("ornek_video.mp4") as reader:
        ...     print(f"FPS: {reader.fps}, Toplam Kare: {reader.total_frames}")
        ...     for frame_rgb in reader.iter_frames(max_frames=10):
        ...         print(frame_rgb.shape)

    Note:
        Daha basit jeneratör akışları ve zaman damgası takibi için modül düzeyindeki
        :func:`read_video_stream` fonksiyonu tercih edilmelidir.
    """

    def __init__(self, path: str | Path) -> None:
        """Video dosyasını açar ve video akışı meta verilerini inceler.

        Args:
            path (str | Path): Açılacak video dosyasının dosya sistemi yolu.

        Raises:
            FileNotFoundError: Belirtilen video dosyası bulunamazsa.
            av.AVError: Video dosyası bozuksa veya çözülemiyorsa.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video bulunamadı: {self.path}")

        # PyAV konteyneri açık tutulur; close() veya __exit__ ile kapatılmalıdır.
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        self._total_frames: int = self._estimate_total_frames()
        # average_rate yoksa (bazı VFR videolarda) güvenli varsayılan olarak 25.0 fps kullanılır.
        self._fps: float = float(self.stream.average_rate) if self.stream.average_rate else 25.0

    # ------------------------------------------------------------------
    # Kare Sayısı Tahmini — Üç Kademeli Güvenli Yöntem
    # ------------------------------------------------------------------

    def _estimate_total_frames(self) -> int:
        """Konteynerden toplam kare sayısını 3 kademeli yöntemle hesaplar/tahmin eder.

        Yöntem 1: Konteyner meta verisi (`stream.frames`). En hızlıdır ancak her
                  konteynerde doğru yazılmış olmayabilir.
        Yöntem 2: Süre ve FPS çarpımı (`duration * fps`). Meta veride süre varsa kullanılır.
        Yöntem 3: Akışı baştan sona decode ederek sayma. En yavaş ama en kesin çözümdür,
                  son çare olarak başvurulur.

        Returns:
            int: Videodaki tahmini veya kesin toplam kare sayısı (asgari 1).
        """
        # 1. Yöntem: Konteyner meta verisi
        total = self.stream.frames
        if total and total > 0:
            return total

        # 2. Yöntem: duration * fps hesabı
        duration = self.stream.duration
        time_base = self.stream.time_base
        avg_rate = self.stream.average_rate
        if duration and time_base and avg_rate:
            return int(float(duration * time_base) * float(avg_rate))

        # 3. Yöntem: Kareleri decode ederek say, ardından konteyneri başa sar
        total = sum(1 for _ in self.container.decode(video=0))
        self.container.seek(0)
        return total or 1

    @property
    def total_frames(self) -> int:
        """Videodaki tahmini veya kesin toplam kare sayısı."""
        return self._total_frames

    @property
    def fps(self) -> float:
        """Videonun saniye başına ortalama kare hızı (Frames Per Second)."""
        return self._fps

    def duration_seconds(self) -> float:
        """Videonun toplam süresini saniye cinsinden hesaplar.

        Returns:
            float: Saniye cinsinden video süresi.
        """
        return self._total_frames / self._fps if self._fps else 0.0

    def iter_frames(self, max_frames: int | None = None) -> Iterator[NDArray[np.uint8]]:
        """RGB24 formatında numpy dizisi olarak kare piksellerini döner.

        Meta veriler (zaman damgası, kare indeksi) gerekmeksizin yalnızca piksel
        matrisleri üzerinden hızlı tarama yapılmak istendiğinde kullanılır.

        Args:
            max_frames (int | None, optional): Okunacak azami kare sayısı.
                None verilirse videonun sonuna kadar okur. Varsayılan None.

        Yields:
            NDArray[np.uint8]: (H, W, 3) boyutlu uint8 formatında RGB piksel dizisi.
        """
        self.container.seek(0)
        count = 0
        for frame in self.container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")
            count += 1
            if max_frames is not None and count >= max_frames:
                break

    def close(self) -> None:
        """Açık olan PyAV konteynerini kapatır ve dosya/bellek tanıtıcılarını serbest bırakır."""
        self.container.close()

    def __enter__(self) -> "VideoReader":
        """Context manager giriş noktası."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager çıkışında otomatik kaynak temizliği."""
        self.close()


# ---------------------------------------------------------------------------
# read_video_stream — Ön İşleme Katmanının Birincil Giriş Fonksiyonu
# ---------------------------------------------------------------------------

def read_video_stream(video_path: str) -> Iterator[RawFrame]:
    """Videoyu baştan sona akış (generator) halinde çözer; her kare için RawFrame üretir.

    Kanal B ön işleme katmanı (:mod:`preprocessing`) bu fonksiyonu şu şekilde tüketir:

    .. code-block:: python

        for frame in read_video_stream(video_path):
            if frame.frame_index % CANDIDATE_STRIDE != 0:
                continue  # Sadece belirli adımlardaki kareleri işle
            # frame.timestamp_sec ve frame.rgb ile devam et

    Tasarım Özellikleri:
        - **Düşük Bellek İzi:** Kareler gerektikçe decode edilir, RAM'e tüm video dolmaz.
        - **Hassas PTS:** `pts * time_base` formülüyle her kareye tam zaman damgası atanır.
        - **Bozuk Konteyner Fallback:** PTS bilgisi eksikse `frame_index / fps` üzerinden
          güvenli zaman tahmini yapılır.

    Args:
        video_path (str): İncelenecek video dosyasının dosya yolu.

    Yields:
        RawFrame: Sıra indeksi, mutlak zaman damgası ve (H, W, 3) RGB numpy matrisi.

    Raises:
        FileNotFoundError: Belirtilen video dosyası mevcut değilse.
        av.AVError: FFmpeg video konteynerini açamaz veya çözemezse.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video bulunamadı: {path}")

    with av.open(str(path)) as container:
        stream = container.streams.video[0]

        # time_base: PyAV'ın PTS birimini saniyeye çeviren kesir (ör. 1/90000)
        time_base = float(stream.time_base) if stream.time_base else 1.0

        frame_index = 0
        for av_frame in container.decode(video=0):
            # PTS mevcut değilse (bazı bozuk/eksik container'lar) indeks/fps kullan
            if av_frame.pts is not None:
                timestamp_sec = float(av_frame.pts) * time_base
            else:
                fps = float(stream.average_rate) if stream.average_rate else 25.0
                timestamp_sec = frame_index / fps

            # RGB24 formatında numpy array — preprocessing.py cv2 ile işler
            rgb = av_frame.to_ndarray(format="rgb24")

            yield RawFrame(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                rgb=rgb,
            )
            frame_index += 1
