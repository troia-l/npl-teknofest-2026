"""
Kanal B - Uçtan Uca Pipeline Orkestratörü (pipeline.py)
======================================================

Bu modül, Kanal B mimarisinin ana orkestrasyon (yürütme) katmanıdır. Karar Ajanı
(:mod:`src.reasoning.decision_agent`) ve ana sistem (:mod:`src.main`), Kanal B
analizini bu modül üzerindeki :func:`run_channel_b` fonksiyonunu çağırarak başlatır.

İki Çalışma Modu (Çift Modlu Mimari):
--------------------------------------
Girdi videosunun süre ve çözünürlük özelliklerine göre sistem iki moddan birini
otomatik olarak seçer:

1. **Tek Parça Modu (Single-Shot Mode):**
   - **Kriter:** Video süresi <= 60 saniye ve video yüksekliği <= 720p.
   - **İşleyiş:** Video üzerinde yeniden kodlama (transcoding) yapılmaz. Orijinal
     dosya baytları doğrudan VLM modeline iletilir.
   - **Ön Ek Önbelleği (Prefix Caching):** Dosya baytları değişmediği için aynı video
     üzerinden yöneltilen takip soruları (follow-up queries) sunucu tarafında
     ön ek önbelleğinden (prefix cache) yararlanır ve çıkarım hızı katlanarak artar.

2. **İterasyonlu Sahne Bazlı Mod (Iterative Scene-Based Mode):**
   - **Kriter:** Video süresi > 60 saniye veya çözünürlük > 720p.
   - **İşleyiş:** Video, :func:`preprocessing.detect_scene_boundaries` ile sahne
     geçiş noktalarından en fazla 60 saniyelik parçalara bölünür ve 720p H.264
     formatında optimize edilir.
   - **Metin Tabanlı Dinamik Hafıza:** Her segment sırayla işlenir. Segment çıktısı
     anlamsal ve kronolojik olarak damıtılarak :class:`contracts.VideoAnalysisMemory`
     yapısına kaydedilir ve sonraki segmentin promptuna aktarılır.
   - **Doğrusal Bağlam Maliyeti Önleme:** Ham çıktıların biriktirilmesi yerine
     damıtılmış metin hafıza kullanıldığı için token bütçesi segment sayısıyla
     doğrusal büyümez, sabit bir tavanda kalır.
   - **Mutlak Zaman Ekseni:** Her segment analiz edilirken tam videoya göre mutlak
     başlangıç zamanı modele aktarılır.

Veri Akış Diyagramı:
--------------------
    [Karar Ajanı / API]
            │
            ▼
     run_channel_b()
            │
            ├──► probe_video() & prepare_video_segments()
            │       ├── [Kısa Video] ──► Tek Klip (Yeniden Kodlama Yok)
            │       └── [Uzun Video] ──► Sahne Bölümleme (720p/60sn Klipler)
            │
            ├──► build_backend(VLM_CONFIG)
            │       ├── EVREN Server Backend (varsayılan)
            │       └── Local Fallback (Grid / Mozaik Modu: vLLM, llama.cpp)
            │
            └──► infer_video() Döngüsü + VideoAnalysisMemory
                    │
                    ▼
            S8 JSON Sözlüğü (VLMInterpretation / BatchResult)
"""

from __future__ import annotations

import json
import os
from typing import Any

# Paket içi veya doğrudan dizin çalıştırma uyumlu esnek importlar
try:
    from .contracts import (
        BatchResult,
        SegmentResult,
        VideoAnalysisMemory,
    )
    from .preprocessing import (
        build_vlm_frame_packet,
        detect_scene_boundaries,
        prepare_video_segments,
        probe_video,
    )
    from .vlm_backend import build_backend
except ImportError:
    from contracts import (
        BatchResult,
        SegmentResult,
        VideoAnalysisMemory,
    )
    from preprocessing import (
        build_vlm_frame_packet,
        detect_scene_boundaries,
        prepare_video_segments,
        probe_video,
    )
    from vlm_backend import build_backend


# ---------------------------------------------------------------------------
# Konfigürasyon Yönetimi
# ---------------------------------------------------------------------------

def _load_vlm_config() -> dict[str, Any]:
    """Proje kökündeki `config.yaml` dosyasından VLM arka yüz ayarlarını yükler.

    Dosya bulunamazsa veya yapılandırma eksikse, TEKNOFEST EVREN VLM servisi için
    tanımlı güvenli varsayılan değerleri döndürür.

    Desteklenen Arka Yüzler (Backends):
        - `server`: OpenAI-uyumlu harici servis (TEKNOFEST EVREN VLM). Varsayılandır.
        - `vllm`: Yerel vLLM HTTP API servisi.
        - `llama.cpp`: Yerel llama-server HTTP API servisi.
        - `transformers`: Süreç içi doğrudan HuggingFace pipeline yüklemesi.

    Returns:
        dict[str, Any]: :func:`vlm_backend.build_backend` fonksiyonunun beklediği parametreler.
    """
    fallback: dict[str, Any] = {
        "backend": "server",
        "base_url": "https://evren-llmapi.ssyz.org.tr/v1",
        "model_name": "llm-large",
        "video_model": "vlm",
        "api_key_env": "EVREN_API_KEY",
        "max_tokens": 65536,
        "temperature": 0.15,
        "timeout_sec": 1800,
        "enable_thinking": False,
    }

    config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    )
    if not os.path.exists(config_path):
        return fallback

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        vlm_cfg = cfg.get("vlm", {})

        # "auto" seçimi durumunda tercih listesindeki ilk arka yüzü kullan
        backend_choice = vlm_cfg.get("default_backend", "server")
        if backend_choice == "auto":
            backend_choice = (vlm_cfg.get("auto_preference") or ["server"])[0]

        # config.yaml anahtarlarındaki alt çizgiyi ve noktayı normalize et
        backend_key = backend_choice.replace(".", "_")
        spec = vlm_cfg.get(backend_key, {}) or {}

        if backend_choice == "server":
            return {
                "backend": "server",
                "base_url": spec.get("base_url", fallback["base_url"]),
                "model_name": spec.get("model_name", fallback["model_name"]),
                "video_model": spec.get("video_model", fallback["video_model"]),
                "api_key_env": spec.get("api_key_env", fallback["api_key_env"]),
                "max_tokens": spec.get("max_tokens", fallback["max_tokens"]),
                "temperature": spec.get("temperature", fallback["temperature"]),
                "timeout_sec": spec.get("timeout_sec", fallback["timeout_sec"]),
                "enable_thinking": spec.get("enable_thinking", False),
            }

        if backend_choice in ("vllm", "llama.cpp"):
            return {
                "backend": backend_choice,
                "base_url": spec.get("base_url", "http://localhost:8080"),
                "model_name": spec.get("model", "unknown-model"),
                "max_tokens": spec.get("max_tokens", spec.get("max_new_tokens", 1024)),
                "temperature": spec.get("temperature", 0.15),
                "timeout_sec": spec.get("timeout_sec", 1800),
            }

        if backend_choice == "transformers":
            return {
                "backend": "transformers",
                "model_name": spec.get("model", "llava-hf/LLaVA-NeXT-Video-7B-hf"),
                "device": spec.get("device", "cuda"),
                "max_tokens": spec.get("max_new_tokens", 512),
            }

        print(f"[pipeline] Bilinmeyen backend '{backend_choice}'; EVREN'e dönülüyor.")
        return fallback

    except Exception as exc:
        print(f"[pipeline] config.yaml okunamadı ({exc}), EVREN fallback kullanılıyor.")
        return fallback


#: Modül seviyesinde başlatılan global VLM konfigürasyonu
VLM_CONFIG: dict[str, Any] = _load_vlm_config()


# ---------------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# ---------------------------------------------------------------------------

def _write_json(obj: dict[str, Any], path: str) -> None:
    """Verilen sözlüğü UTF-8 kodlamasıyla diske JSON dosyası olarak kaydeder.

    Args:
        obj (dict[str, Any]): Kaydedilecek sözlük nesnesi.
        path (str): Hedef dosya yolu.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _run_grid_fallback(backend: Any, video_path: str, video_id: str, output_dir: str) -> dict[str, Any]:
    """Video akışını doğrudan desteklemeyen yerel modeller için 2x4 grid fallback yolunu çalıştırır.

    Yerel arka yüzler (`vllm`, `llama.cpp`, `transformers`) doğrudan MP4 video
    girdisi kabul etmez. Bu modeller için videodan kritik 8 kare seçilip tek bir
    mozaik görseli (S1b paketi) olarak modele sunulur.

    Args:
        backend (Any): :class:`vlm_backend.VLMBackendBase` türevi model örneği.
        video_path (str): Kaynak video dosya yolu.
        video_id (str): Video kimliği.
        output_dir (str): Çıktıların yazılacağı dizin yolu.

    Returns:
        dict[str, Any]: S8 sözleşmesine uygun sözlük çıktısı.
    """
    print("[pipeline] Backend video modunu desteklemiyor; grid yoluna düşülüyor.")
    packet = build_vlm_frame_packet(video_path, video_id, output_dir)
    interpretation = backend.infer(packet)
    _write_json(
        interpretation.to_dict(),
        os.path.join(output_dir, f"{packet.packet_id}_vlm_interpretation.json"),
    )
    return interpretation.to_dict()


# ---------------------------------------------------------------------------
# Kanal B Ana Giriş Noktası
# ---------------------------------------------------------------------------

def run_channel_b(video_path: str, video_id: str, output_dir: str = "./out/channel_b") -> dict[str, Any]:
    """Kanal B'nin uçtan uca yürütücüsü: Video dosyasını işler ve S8 sözlüğünü döndürür.

    Video süresine ve çözünürlüğüne bağlı olarak otomatik olarak:
      - Süre <= 60 sn ve çözünürlük <= 720p ise Tek Parça Modu ile çıkarım yapar.
      - Aksi halde videoyu sahne sınırlarından segmentlere bölerek İteratif Mod uygular.

    Args:
        video_path (str): Analiz edilecek video dosyasının tam dosya yolu.
        video_id (str): İzlenebilirlik ve denetim kayıtları için benzersiz video kimliği.
        output_dir (str, optional): Segment klipleri ve JSON kayıtlarının yazılacağı dizin.
            Varsayılan: "./out/channel_b".

    Returns:
        dict[str, Any]: Karar Ajanı'nın tüketimine hazır S8 sözleşmesi sözlüğü:
            - `scene_summary_tr`: Sahne özeti.
            - `detected_entities`: Tespit edilen varlıklar.
            - `detected_actions_tr`: Gözlemlenen eylemler.
            - `risk_events`: Yapılandırılmış risk olayları.
            - `risk_flags_tr`: Düz metin risk açıklamaları.
            - `confidence_overall`: Model genel güven skoru.
            - `computed_confidence`: Bağımsız hesaplanan güven derecesi.
            - `segment_count`: İşlenen segment adedi.
            - `failed_segments`: Hata alan segmentlerin listesi.

    Raises:
        FileNotFoundError: Video dosyası mevcut değilse.
        RuntimeError: VLM backend servisiyle iletişim kurulamadığında.
    """
    backend = build_backend(VLM_CONFIG)
    os.makedirs(output_dir, exist_ok=True)

    # Video modunu desteklemeyen yerel modeller için grid fallback
    if not hasattr(backend, "infer_video"):
        return _run_grid_fallback(backend, video_path, video_id, output_dir)

    info = probe_video(video_path)
    clips = prepare_video_segments(video_path, os.path.join(output_dir, "segments"))

    # --- Mod 1: Tek Parça Modu (Kısa ve Uygun Çözünürlüklü Videolar) ---
    if len(clips) == 1:
        clip = clips[0]
        print(
            f"[pipeline] Tek parça mod: {info['width']}x{info['height']}, "
            f"{info['duration_sec']:.1f}sn (yeniden kodlama: {clip.reencoded})"
        )
        interpretation = backend.infer_video(
            video_path=clip.path,
            video_id=video_id,
            clip_start_sec=clip.start_sec,
            clip_end_sec=clip.end_sec,
            is_segment=False,
        )
        _write_json(
            interpretation.to_dict(),
            os.path.join(output_dir, f"{interpretation.packet_id}_vlm_interpretation.json"),
        )
        return interpretation.to_dict()

    # --- Mod 2: İterasyonlu Sahne Bazlı Mod (Uzun Videolar) ---
    print(
        f"[pipeline] İterasyonlu mod: {info['duration_sec']:.1f}sn video "
        f"{len(clips)} segmente bölündü (segment tavanı 60sn / 720p)"
    )

    memory = VideoAnalysisMemory()
    segments: list[SegmentResult] = []
    failed: list[dict[str, Any]] = []

    for clip in clips:
        print(f"[pipeline]   segment {clip.index + 1}/{len(clips)}: {clip.time_label}")
        try:
            interpretation = backend.infer_video(
                video_path=clip.path,
                video_id=video_id,
                clip_start_sec=clip.start_sec,
                clip_end_sec=clip.end_sec,
                memory_context=memory.to_prompt_context(),
                is_segment=True,
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"[pipeline]   UYARI: segment {clip.index} işlenemedi ({err})")
            failed.append({
                "segment_index": clip.index,
                "start_sec": clip.start_sec,
                "end_sec": clip.end_sec,
                "error": err,
            })
            continue

        # Hafızayı güncelle → sonraki segment bu bağlamla yorumlanır
        memory.absorb(interpretation, clip.start_sec)

        segments.append(SegmentResult(
            segment_index=clip.index,
            start_sec=clip.start_sec,
            end_sec=clip.end_sec,
            interpretation=interpretation,
        ))
        _write_json(
            interpretation.to_dict(),
            os.path.join(
                output_dir,
                f"{interpretation.packet_id}_seg{clip.index:03d}_vlm_interpretation.json",
            ),
        )

    batch = BatchResult(
        video_id=video_id,
        segments=segments,
        failed_segments=failed,
        memory_context=memory.to_prompt_context(),
    )
    _write_json(batch.to_dict(), os.path.join(output_dir, f"{video_id}_batch.json"))

    critical = batch.most_critical_segment
    print(
        f"[pipeline] Tamamlandı: {batch.total_segments} segment, "
        f"{batch.total_risk_events} risk olayı, "
        f"{len(failed)} başarısız, en kritik segment: "
        f"{critical.segment_index if critical else 'yok'}"
    )
    return batch.to_interpretation_dict()


# ---------------------------------------------------------------------------
# Sahne Bazlı Batch İşleme Fonksiyonu
# ---------------------------------------------------------------------------

def run_channel_b_scene_based(
    video_path: str,
    video_id: str,
    output_dir: str = "./out/channel_b",
    ssim_threshold: float | None = None,
    min_segment_sec: float | None = None,
    max_segment_sec: float | None = None,
) -> BatchResult:
    """Videoyu segment segment analiz ederek tam :class:`contracts.BatchResult` nesnesi döndürür.

    Sonucu tekil bir sözlüğe indirgemek yerine, her bir segmentin ayrıntılı
    yorumlarını, en kritik segment indeksini ve varsa başarısız segment kayıtlarını
    içeren kapsamlı nesneyi korur. Kullanıcı arayüzü (UI) zaman çizelgesi veya
    ayrıntılı denetim raporları için tercih edilir.

    Args:
        video_path (str): İncelenecek video dosyası.
        video_id (str): Video kimliği.
        output_dir (str, optional): Çıktı dizini. Varsayılan: "./out/channel_b".
        ssim_threshold (float | None, optional): Sahne kesim SSIM eşiği (None ise config: 0.30).
        min_segment_sec (float | None, optional): Asgari segment süresi (None ise config: 15.0).
        max_segment_sec (float | None, optional): Azami segment süresi (None ise config: 60.0).

    Returns:
        BatchResult: Tüm segment sonuçlarını ve özet istatistikleri içeren nesne.

    Raises:
        RuntimeError: Arka yüz doğrudan video çıkarımını desteklemiyorsa.
    """
    backend = build_backend(VLM_CONFIG)
    if not hasattr(backend, "infer_video"):
        raise RuntimeError(
            "Segment bazlı analiz video destekli bir backend gerektirir "
            "(config.yaml → vlm.default_backend: server)."
        )

    os.makedirs(output_dir, exist_ok=True)
    clips = prepare_video_segments(
        video_path,
        os.path.join(output_dir, "segments"),
        max_segment_sec=max_segment_sec,
        ssim_threshold=ssim_threshold,
        min_segment_sec=min_segment_sec,
    )
    print(f"[pipeline] {len(clips)} segment işlenecek: {video_id}")

    memory = VideoAnalysisMemory()
    segments: list[SegmentResult] = []
    failed: list[dict[str, Any]] = []

    for clip in clips:
        print(
            f"[pipeline] Segment {clip.index + 1}/{len(clips)}: "
            f"{clip.start_sec:.1f}sn – {clip.end_sec:.1f}sn"
        )
        try:
            interpretation = backend.infer_video(
                video_path=clip.path,
                video_id=video_id,
                clip_start_sec=clip.start_sec,
                clip_end_sec=clip.end_sec,
                memory_context=memory.to_prompt_context(),
                is_segment=len(clips) > 1,
            )
            memory.absorb(interpretation, clip.start_sec)
            segments.append(SegmentResult(
                segment_index=clip.index,
                start_sec=clip.start_sec,
                end_sec=clip.end_sec,
                interpretation=interpretation,
            ))
            _write_json(
                interpretation.to_dict(),
                os.path.join(
                    output_dir,
                    f"{interpretation.packet_id}_seg{clip.index:03d}_vlm_interpretation.json",
                ),
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"[pipeline] UYARI: Segment {clip.index} işlenemedi ({err}), atlanıyor.")
            failed.append({
                "segment_index": clip.index,
                "start_sec": clip.start_sec,
                "end_sec": clip.end_sec,
                "error": err,
            })

    result = BatchResult(
        video_id=video_id,
        segments=segments,
        failed_segments=failed,
        memory_context=memory.to_prompt_context(),
    )
    _write_json(result.to_dict(), os.path.join(output_dir, f"{video_id}_batch.json"))

    critical = result.most_critical_segment
    print(
        f"[pipeline] Tamamlandı: {result.total_segments} segment, "
        f"{result.total_risk_events} risk olayı, "
        f"en kritik segment: {critical.segment_index if critical else 'yok'}"
    )
    return result


# ---------------------------------------------------------------------------
# Doğrudan Komut Satırı Test Bloğu
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    video = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
    out = run_channel_b(video, video_id=os.path.splitext(os.path.basename(video))[0])
    print(json.dumps(out, ensure_ascii=False, indent=2)[:3000])
