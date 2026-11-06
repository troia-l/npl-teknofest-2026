"""
Kanal B (Channel B) - Görsel Dil Modeli (VLM) Video & Sahne Analiz Katmanı
==========================================================================

Bu paket, TEKNOFEST Yapay Zeka Yarışması kapsamında Kanal B (semantik sahne analizi
ve bağımsız görsel algı) bileşenlerini içerir.

Mimari Özeti:
-------------
Kanal B, kural tabanlı ve nesne tespitine dayalı Kanal A'dan bağımsız olarak çalışır.
Videonun tamamını veya kilit anlarını multimodal Görsel Dil Modelleri (VLM) ile
yorumlayarak sahne özeti, eylemler, varlıklar ve iş sağlığı ve güvenliği (İSG) risklerini
yapılandırılmış JSON (S8 Sözleşmesi) formatında Karar Ajanı'na (Decision Agent) iletir.

Desteklenen Çıkarım Modları:
---------------------------
1. **Video Modu (Doğrudan Çıkarım)**:
   - TEKNOFEST EVREN VLM servisi üzerinden yerel 720p / 60 sn klipler doğrudan gönderilir.
   - 60 saniyeden uzun videolarda sahne değişimlerine göre parçalama ve metin tabanlı
     zamansal/anlamsal hafıza (:class:`contracts.VideoAnalysisMemory`) ile iteratif analiz yapılır.

2. **Grid / Mozaik Modu (Geriye Dönük / Yerel Modeller)**:
   - Yerel sunucular (vLLM, llama.cpp, Transformers) için videodan en kritik ve keskin
     8 kare (SSIM & Laplacian) seçilerek 2x4 ızgara (S1b Paketi) haline getirilir.

Ana Modüller:
-------------
- :mod:`pipeline`: Kanal B'nin uçtan uca çalıştırıcısı (Orchestrator).
- :mod:`contracts`: S1b (Girdi) ve S8 (Çıktı) veri sözleşmeleri ve hafıza yapıları.
- :mod:`vlm_backend`: EVREN Server, vLLM, llama.cpp ve HuggingFace adaptörleri.
- :mod:`preprocessing`: Sahne kesimi, SSIM analizi, CLAHE iyileştirmesi ve grid montajı.
- :mod:`video_reader`: PyAV tabanlı yüksek başarımlı ve zaman damgası (PTS) korumalı video okuyucu.
"""

from __future__ import annotations

from .contracts import (
    BatchResult,
    DetectedEntity,
    EnhancementInfo,
    FrameMeta,
    FrameQualityMetrics,
    GridLayout,
    InferenceMetrics,
    RiskEvent,
    SegmentResult,
    SEVERITY_RANK,
    VideoAnalysisMemory,
    VLMFramePacket,
    VLMInterpretation,
    format_mmss,
)
from .pipeline import (
    run_channel_b,
    run_channel_b_scene_based,
)
from .preprocessing import (
    VideoClip,
    build_vlm_frame_packet,
    detect_scene_boundaries,
    extract_video_segment,
    prepare_video_segments,
    probe_video,
)
from .video_reader import (
    RawFrame,
    VideoReader,
    read_video_stream,
)
from .vlm_backend import (
    LlamaCppBackend,
    ServerBackend,
    TransformersBackend,
    VLLMBackend,
    VLMBackendBase,
    build_backend,
    build_system_prompt,
    build_video_system_prompt,
)

__all__ = [
    # Pipeline Orchestration
    "run_channel_b",
    "run_channel_b_scene_based",
    # Data Contracts (S1b & S8)
    "VLMFramePacket",
    "VLMInterpretation",
    "RiskEvent",
    "DetectedEntity",
    "InferenceMetrics",
    "FrameMeta",
    "FrameQualityMetrics",
    "GridLayout",
    "EnhancementInfo",
    "SegmentResult",
    "BatchResult",
    "VideoAnalysisMemory",
    "SEVERITY_RANK",
    "format_mmss",
    # Preprocessing
    "VideoClip",
    "build_vlm_frame_packet",
    "detect_scene_boundaries",
    "extract_video_segment",
    "prepare_video_segments",
    "probe_video",
    # Video Streaming & Reading
    "RawFrame",
    "VideoReader",
    "read_video_stream",
    # VLM Backend Adapters
    "VLMBackendBase",
    "ServerBackend",
    "VLLMBackend",
    "LlamaCppBackend",
    "TransformersBackend",
    "build_backend",
    "build_system_prompt",
    "build_video_system_prompt",
]
