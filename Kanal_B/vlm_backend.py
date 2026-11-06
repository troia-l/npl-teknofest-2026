"""
Kanal B - Görsel Dil Modeli (VLM) Çıkarım Arka Yüzü (vlm_backend.py)
==================================================================

Bu modül, Görsel Dil Modelleri (Vision-Language Models / Multimodal LLM) ile
iletişim kurarak video ve görsel paketlerini semantik sahne yorumlarına ve yapılandırılmış
İSG risk analizlerine (S8 Sözleşmesi) dönüştürür.

Mimari Tasarım ve Desenler:
---------------------------
1. **Adaptör Deseni (Adapter Pattern):**
   - :class:`VLMBackendBase` soyut taban sınıfı, tüm çıkarım motorları için standart bir
     arayüz tanımlar (`infer()` ve `_raw_infer()`).
   - Farklı altyapılar (TEKNOFEST EVREN Server, vLLM, llama.cpp, HuggingFace Transformers)
     ortak arayüz arkasında şeffafça değiştirilebilir.

2. **Çift Mod Desteği (Video ve Grid):**
   - **Video Modu (:meth:`ServerBackend.infer_video`):** TEKNOFEST EVREN `vlm` alias'ı
     üzerinden MP4 video klipleri doğrudan gönderilir. Zamansal akış ve hareket analizi
     yüksek doğrulukla yapılır.
   - **Grid Modu (:meth:`VLMBackendBase.infer`):** Görsel tabanlı yerel modeller için
     kare mozaiği (S1b paketi) üzerinden tek HTTP çağrısıyla çıkarım sağlanır.

3. **3 Kademeli Güvenli JSON Ayrıştırıcı ve Düşünce İzi Ayrımı:**
   - Qwen2.5-VL gibi akıl yürüten (reasoning) modellerin `<think>...</think>` blokları
     ayrıştırılarak `reasoning_trace` alanına aktarılır; JSON gövdesinden temizlenir.
   - Model çıktısı tam JSON, Markdown blokları (```json) veya greedy süslü parantez
     aramasıyla 3 kademede çözülür. Hata durumunda sistem çökmez, S8 sözleşmesi korunur.

4. **Bağımsız Güven Skoru (:func:`_compute_confidence`):**
   - Yalnızca modelin kendi öz güven beyanına (`confidence_overall`) güvenmek yerine;
     parse başarısı, risk şiddeti ağırlıklandırması ve görsel kanıt sayısı
     (supporting frame count) birleştirilerek kalibre edilmiş bir güven skoru üretilir.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import requests

# Paket içi veya doğrudan dizin çalıştırma uyumlu esnek importlar
try:
    from .contracts import (
        DetectedEntity,
        InferenceMetrics,
        RiskEvent,
        VLMFramePacket,
        VLMInterpretation,
        format_mmss,
    )
except ImportError:
    from contracts import (
        DetectedEntity,
        InferenceMetrics,
        RiskEvent,
        VLMFramePacket,
        VLMInterpretation,
        format_mmss,
    )


# ---------------------------------------------------------------------------
# Grid Modu Sistem Promptu Üreticisi
# ---------------------------------------------------------------------------

def build_system_prompt(rows: int, cols: int) -> str:
    """Grid matris boyutlarına göre dinamik VLM sistem promptu üretir.

    Hücre sayısını ve indeks sınırlarını (`rows * cols`) dinamik olarak enjekte
    ederek modelin doğru ızgara konumlarına atıfta bulunmasını sağlar.

    Args:
        rows (int): Grid görselindeki satır adedi.
        cols (int): Grid görselindeki sütun adedi.

    Returns:
        str: Modele gönderilecek İngilizce sistem promptu metni.
    """
    total_cells = rows * cols
    max_idx = total_cells - 1
    return (
        "You are a security camera scene analyst. You are given a single grid image "
        "containing frames taken from a video stream. "
        f"The grid is {rows}×{cols}, cells are numbered left-to-right, top-to-bottom "
        f"starting from 0 (cells 0-{max_idx}, {total_cells} total). "
        "Your task is to interpret the scene with your own eyes, INDEPENDENTLY of the "
        "event engine and the object detector.\n\n"
        "VEHICLE IDENTIFICATION — THINK STEP BY STEP:\n"
        "When you see ANY vehicle or machine, reason about its exact type before labeling it. "
        "Consider visual cues: size, shape, wheels vs tracks, cabin position, forks, boom arm, "
        "bucket, flatbed, road context vs industrial site. Possible types include but are not "
        "limited to: forklift, crane, excavator, loader, truck, pickup, car, van, bus, "
        "motorcycle, bicycle. A vehicle on a public road is most likely a car/truck/bus — "
        "do NOT assume it is industrial equipment. Write your reasoning in the "
        "\"vehicle_type_reasoning\" field.\n\n"
        "Produce an output that matches ONLY the following JSON schema and "
        "contains no other text:\n"
        "{\n"
        '  "scene_summary_tr": "1-3 sentence scene summary, in English",\n'
        '  "vehicle_type_reasoning": "Step-by-step reasoning about what type each vehicle/machine is and why",\n'
        '  "detected_entities": [{"label": "specific vehicle type or object label", "confidence_hint": "low|medium|high", "notes_tr": "short note, in English"}],\n'
        '  "detected_actions_tr": ["observed actions, in English"],\n'
        '  "risk_events": [\n'
        '    {\n'
        '      "description_tr": "risk description, in English",\n'
        '      "severity": "low|medium|high|critical",\n'
        '      "confidence": 0.85,\n'
        '      "supporting_frame_count": 2,\n'
        '      "supporting_frame_positions": [1, 4]\n'
        '    }\n'
        '  ],\n'
        '  "confidence_overall": 0.0\n'
        "}\n"
        "risk_events RULES: return an empty list [] if there is no risk. "
        "severity: low=low priority, medium=attention, high=significant risk, critical=human health in danger. "
        f"supporting_frame_count: in how many grid cells you saw this risk (0-{total_cells}). "
        f"supporting_frame_positions: grid indices where the risk is visible (0=top-left, left-to-right, max={max_idx}). "
        "confidence: your risk-specific confidence (0.0-1.0)."
    )


# ---------------------------------------------------------------------------
# Yardımcı Fonksiyonlar ve JSON / Reasoning Ayrıştırıcılar
# ---------------------------------------------------------------------------

def _encode_jpeg_b64(image_path: str) -> str:
    """Diskteki JPEG grid görselini Base64 formatına kodlar.

    Args:
        image_path (str): Görselin mutlak dosya yolu.

    Returns:
        str: Base64 kodlanmış ASCII metin.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_json(text: str) -> dict[str, Any]:
    """Modelin metin çıktısından JSON bloğunu 3 kademeli stratejiyle ayıklar ve parse eder.

    Kademeler:
        1. Kademe: Doğrudan `json.loads()` (en temiz durum).
        2. Kademe: Markdown kod bloğu (` ```json { ... } ``` `) regex yakalama.
        3. Kademe: Metindeki en dış `{ ... }` süslü parantezlerini greedy yakalama.

    Args:
        text (str): Modelden gelen ham metin (reasoning etiketleri temizlenmiş).

    Returns:
        dict[str, Any]: Ayrıştırılmış JSON sözlüğü.

    Raises:
        ValueError: Hiçbir kademede geçerli bir JSON sözlüğü bulunamazsa.
    """
    stripped = text.strip()

    # 1. Kademe: Direkt parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2. Kademe: Markdown fence içinde ara
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Kademe: İlk { ... } bloğunu yakala
    brace = re.search(r"\{.*\}", stripped, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"VLM çıktısında geçerli JSON bulunamadı: {text[:200]}")


def _extract_reasoning_and_json(text: str) -> tuple[str, dict[str, Any]]:
    """Ham model yanıtından düşünce izini (<think>) ayırır ve kalan metni JSON olarak ayrıştırır.

    Args:
        text (str): Modelin ham tamamlanma metni.

    Returns:
        tuple[str, dict[str, Any]]: (Ayıklanan düşünce metni, ayrıştırılmış JSON sözlüğü).
    """
    reasoning = ""
    think_match = re.search(
        r"<(?:think|düşünme)>(.*?)</(?:think|düşünme)>",
        text,
        re.DOTALL,
    )
    if think_match:
        reasoning = think_match.group(1).strip()
        text = text[: think_match.start()] + text[think_match.end() :]
    return reasoning, _extract_json(text.strip())


def _compute_confidence(parsed: dict[str, Any], parse_succeeded: bool) -> float:
    """Modelin öz güveninden bağımsız, kalibre edilmiş çok etmenli güven derecesi hesaplar.

    Formül Bileşenleri:
        - Parse Başarısı: JSON bozuksa güven anında 0.0 döner.
        - Model Öz Güveni: %50 ağırlıkla formüle girer (`confidence_overall * 0.5`).
        - Risk Tutarlılığı: Tespit edilen risklerin şiddet ve güven ağırlıklı ortalaması (%30 ağırlık).
        - Kanıt Çeşitliliği: Riskin görüldüğü ortalama kare sayısına bağlı bonus (azami +0.2).

    Args:
        parsed (dict[str, Any]): Ayrıştırılmış model yanıt sözlüğü.
        parse_succeeded (bool): JSON ayrıştırmanın başarılı olup olmadığı.

    Returns:
        float: 0.0 ile 1.0 aralığında yuvarlanmış güven katsayısı.
    """
    if not parse_succeeded:
        return 0.0

    model_conf = float(parsed.get("confidence_overall", 0.0) or 0.0)

    raw_risks = [r for r in parsed.get("risk_events", []) if isinstance(r, dict)]
    if raw_risks:
        _severity_w = {"low": 0.25, "medium": 0.50, "high": 0.75, "critical": 1.00}
        weighted = [
            float(r.get("confidence", 0.5))
            * _severity_w.get(r.get("severity", "low"), 0.25)
            for r in raw_risks
        ]
        risk_score = min(sum(weighted) / max(len(weighted), 1), 1.0)
        avg_frames = (
            sum(int(r.get("supporting_frame_count", 1)) for r in raw_risks)
            / len(raw_risks)
        )
        evidence_bonus = min(avg_frames / 8.0, 1.0) * 0.2
    else:
        risk_score = 0.0
        evidence_bonus = 0.0

    computed = (model_conf * 0.5) + (risk_score * 0.3) + evidence_bonus
    return round(min(1.0, max(0.0, computed)), 4)


# ---------------------------------------------------------------------------
# Soyut Taban Sınıf (VLM Backend Arayüzü)
# ---------------------------------------------------------------------------

class VLMBackendBase(ABC):
    """Tüm VLM çıkarım motorları için temel arayüz sınıfı.

    Tüm backend sınıfları bu sınıftan türer ve :meth:`_raw_infer` metodunu kendi
    iletişim protokolüne (HTTP REST, C++ bağlamı, Transformers pipeline) göre uygular.
    Ortak :meth:`infer` şablon metodu gecikme ölçümü, JSON doğrulama ve S8 paketlemesini yönetir.
    """
    backend_name: str
    model_name: str

    @abstractmethod
    def _raw_infer(self, image_path: str, prompt: str) -> tuple[str, int]:
        """Görsel ve promptu modele ileten soyut metod.

        Args:
            image_path (str): Gönderilecek görselin disk yolu.
            prompt (str): Modele iletilecek sistem ve kullanıcı talimatları.

        Returns:
            tuple[str, int]: (Model ham metin çıktısı, üretilen token adedi).
        """

    def infer(self, packet: VLMFramePacket) -> VLMInterpretation:
        """S1b paketini alır, çıkarım yapar ve S8 :class:`contracts.VLMInterpretation` üretir.

        Args:
            packet (VLMFramePacket): Giriş S1b veri paketi.

        Returns:
            VLMInterpretation: Karar Ajanı için yapılandırılmış S8 yorum paketi.
        """
        system_prompt = build_system_prompt(
            packet.grid_layout.rows, packet.grid_layout.cols
        )

        t0 = time.perf_counter()
        raw_text, tokens = self._raw_infer(packet.grid_image_path, system_prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        parse_succeeded = True
        reasoning_trace = ""
        try:
            reasoning_trace, parsed = _extract_reasoning_and_json(raw_text)
        except ValueError:
            parse_succeeded = False
            parsed = {
                "scene_summary_tr": "Model çıktısı ayrıştırılamadı.",
                "detected_entities": [],
                "detected_actions_tr": [],
                "risk_events": [],
                "confidence_overall": 0.0,
            }

        entities = [
            DetectedEntity(
                label=e.get("label", ""),
                confidence_hint=e.get("confidence_hint", "low"),
                notes_tr=e.get("notes_tr", ""),
            )
            for e in parsed.get("detected_entities", [])
            if isinstance(e, dict)
        ]

        risk_events = [
            RiskEvent(
                description_tr=r.get("description_tr", ""),
                severity=r.get("severity", "low"),
                confidence=float(r.get("confidence", 0.5)),
                supporting_frame_count=int(r.get("supporting_frame_count", 1)),
                supporting_frame_positions=r.get("supporting_frame_positions", []),
            )
            for r in parsed.get("risk_events", [])
            if isinstance(r, dict)
        ]

        computed_conf = _compute_confidence(parsed, parse_succeeded)

        return VLMInterpretation(
            packet_id=packet.packet_id,
            video_id=packet.video_id,
            model_name=self.model_name,
            model_backend=self.backend_name,  # type: ignore[arg-type]
            scene_summary_tr=parsed.get("scene_summary_tr", ""),
            detected_entities=entities,
            detected_actions_tr=parsed.get("detected_actions_tr", []),
            risk_events=risk_events,
            confidence_overall=float(parsed.get("confidence_overall", 0.0) or 0.0),
            computed_confidence=computed_conf,
            inference=InferenceMetrics(latency_ms=latency_ms, tokens_generated=tokens),
            raw_model_output=raw_text,
            reasoning_trace=reasoning_trace,
        )


# ---------------------------------------------------------------------------
# Backend 1 — vLLM (OpenAI Uyumlu GPU Sunucusu)
# ---------------------------------------------------------------------------

class VLLMBackend(VLMBackendBase):
    """vLLM OpenAI-uyumlu yerel sunucu üzerinden çıkarım yapan arka yüz.

    Attributes:
        base_url (str): Sunucu kök URL'i (örn. 'http://localhost:8000').
        model_name (str): Sunucuda barındırılan model adı (örn. 'Qwen/Qwen2.5-VL-7B-Instruct').
        max_tokens (int): Üretilecek azami token bütçesi.
        temperature (float): Örnekleme sıcaklığı.
        timeout_sec (int): HTTP istek zaman aşımı (sn).
    """
    backend_name = "vllm"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout_sec: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_sec = timeout_sec

    def _raw_infer(self, image_path: str, prompt: str) -> tuple[str, int]:
        b64 = _encode_jpeg_b64(image_path)
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
        }
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return text, tokens


# ---------------------------------------------------------------------------
# Backend 2 — llama.cpp (GGUF Multimodal Sunucu)
# ---------------------------------------------------------------------------

class LlamaCppBackend(VLMBackendBase):
    """llama-server GGUF multimodal çıkarım motoru adaptörü.

    Attributes:
        base_url (str): Sunucu kök adresi.
        model_name (str): Model adı.
        max_tokens (int): Azami token bütçesi.
        temperature (float): Sıcaklık değeri.
        timeout_sec (int): HTTP zaman aşımı.
    """
    backend_name = "llama.cpp"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout_sec: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_sec = timeout_sec

    def _raw_infer(self, image_path: str, prompt: str) -> tuple[str, int]:
        b64 = _encode_jpeg_b64(image_path)
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
        }
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return text, tokens


# ---------------------------------------------------------------------------
# Backend 3 — HuggingFace Transformers (Süreç İçi Model Yükleme)
# ---------------------------------------------------------------------------

class TransformersBackend(VLMBackendBase):
    """HuggingFace transformers kütüphanesi üzerinden GPU/CPU üzerinde doğrudan model yürütücüsü.

    Lazy loading tasarımıyla model ağırlıkları ilk çıkarım isteğinde belleğe yüklenir.
    """
    backend_name = "transformers"

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._device = device

    def _lazy_load(self) -> None:
        """Model ağırlıklarını ve işlemcisini talep anında (lazy) belleğe yükler."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map=self._device
        )

    def _raw_infer(self, image_path: str, prompt: str) -> tuple[str, int]:
        self._lazy_load()
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        chat_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=chat_prompt, images=image, return_tensors="pt").to(self._device)
        output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        text = self._processor.decode(output[0], skip_special_tokens=True)
        tokens_generated = output.shape[-1] - inputs["input_ids"].shape[-1]
        return text, int(tokens_generated)


# ---------------------------------------------------------------------------
# Backend Fabrika Fonksiyonu
# ---------------------------------------------------------------------------

def build_backend(config: dict[str, Any]) -> VLMBackendBase:
    """Yapılandırma sözlüğüne göre ilgili VLM backend nesnesini oluşturur.

    Args:
        config (dict[str, Any]): `config.yaml` dosyasından okunan veya varsayılan ayarlara
            sahip konfigürasyon sözlüğü.

    Returns:
        VLMBackendBase: Yapılandırılmış arka yüz nesnesi.

    Raises:
        ValueError: Bilinmeyen bir backend adı belirtildiğinde.
    """
    backend = config["backend"]

    if backend == "server":
        api_key = os.environ.get(config.get("api_key_env", "EVREN_API_KEY"), "").strip()
        if not api_key:
            print(
                f"[backend] UYARI: '{config.get('api_key_env', 'EVREN_API_KEY')}' "
                f"ortam değişkeni boş; istek kimlik doğrulaması olmadan gidecek."
            )
        return ServerBackend(
            base_url=config["base_url"],
            model_name=config["model_name"],
            video_model=config.get("video_model"),
            api_key=api_key or None,
            max_tokens=config.get("max_tokens", 65536),
            temperature=config.get("temperature", 0.15),
            timeout_sec=config.get("timeout_sec", 1800),
            enable_thinking=config.get("enable_thinking", False),
        )

    if backend == "vllm":
        return VLLMBackend(
            config["base_url"],
            config["model_name"],
            config.get("max_tokens", 512),
            config.get("temperature", 0.15),
            config.get("timeout_sec", 60),
        )

    if backend == "llama.cpp":
        return LlamaCppBackend(
            config["base_url"],
            config["model_name"],
            config.get("max_tokens", 512),
            config.get("temperature", 0.15),
            config.get("timeout_sec", 60),
        )

    if backend == "transformers":
        return TransformersBackend(
            config["model_name"],
            config.get("device", "cuda"),
            config.get("max_tokens", 512),
        )

    raise ValueError(
        f"Bilinmeyen backend: {backend!r} — 'server', 'vllm', 'llama.cpp' "
        f"veya 'transformers' olmalı"
    )


# ===========================================================================
# TEKNOFEST EVREN SERVER BACKEND — Video ve Görsel Destekli REST Servis
# ===========================================================================

#: EVREN REST API 256 MB gövde sınırına göre Base64 öncesi azami ham dosya boyutu (~190 MB)
_RAW_VIDEO_LIMIT_BYTES: int = int(256 * 1024 * 1024 / 1.34)


def build_video_system_prompt(
    clip_start_sec: float,
    clip_end_sec: float,
    memory_context: str = "",
    is_segment: bool = False,
) -> str:
    """Video modu için mutlak zaman ekseni ve hafıza bağlamı içeren sistem promptunu oluşturur.

    Önemli Kurallar:
        - **Mutlak Zaman Ekseni:** Klibin göreli 0:00 anı yerine videonun gerçek başlangıç
          saniyesi (`clip_start_sec`) modele bildirilir. Risk olaylarının `timestamp_sec`
          alanı bu mutlak saniye aralığında talep edilir.
        - **Metin Hafıza Bağlamı:** Önceki segmentlerden aktarılan varlık ve olaylar
          özetlenerek modele verilir.
        - **Genel Varlık Terminolojisi:** Modelden nesne tespitini genel etiketlerle
          ('person', 'vehicle', 'machine') yapması istenir.

    Args:
        clip_start_sec (float): Klibin tam videodaki mutlak başlangıç saniyesi.
        clip_end_sec (float): Klibin tam videodaki mutlak bitiş saniyesi.
        memory_context (str, optional): Önceki segmentlerden taşınan metin hafızası.
        is_segment (bool, optional): Klibin daha uzun bir videonun segmenti olup olmadığı.

    Returns:
        str: Modele gönderilecek İngilizce/Türkçe sistem promptu metni.
    """
    start_label = format_mmss(clip_start_sec)
    end_label = format_mmss(clip_end_sec)

    if is_segment:
        time_rules = (
            f"IMPORTANT — ABSOLUTE TIME AXIS: This clip is a SEGMENT of a longer "
            f"recording. The clip's own 00:00 corresponds to {start_label} "
            f"({clip_start_sec:.1f} seconds) of the FULL video. The segment covers "
            f"{start_label}-{end_label} of the full video.\n"
            f"Every timestamp you report in \"timestamp_sec\" MUST be in ABSOLUTE "
            f"full-video seconds, i.e. a number between {clip_start_sec:.1f} and "
            f"{clip_end_sec:.1f}. Do NOT report time relative to the clip start.\n\n"
        )
    else:
        time_rules = (
            f"The clip covers the whole video, {start_label}-{end_label} "
            f"(0.0 to {clip_end_sec:.1f} seconds).\n"
            f"Report \"timestamp_sec\" as a number of seconds between 0.0 and "
            f"{clip_end_sec:.1f}.\n\n"
        )

    memory_block = f"{memory_context}\n\n" if memory_context.strip() else ""

    return (
        "You are an occupational health and safety (OHS) video analyst. You are "
        "given a video clip from a work site or surveillance camera. Interpret the "
        "scene with your own eyes, INDEPENDENTLY of any object detector or rule "
        "engine.\n\n"
        + time_rules
        + memory_block
        + "OBJECT NAMING: Use GENERAL terms for what you see: 'person', 'vehicle', "
        "'machine', 'load', 'rack/structure', 'liquid/substance', 'smoke or flame'. "
        "Do not guess specific equipment model names. If a vehicle's type is clearly "
        "visible you may name it, but state your uncertainty in notes_tr.\n\n"
        "Produce output matching ONLY the following JSON schema, with no other text:\n"
        "{\n"
        '  "scene_summary_tr": "1-3 sentence summary of the scene, in Turkish",\n'
        '  "detected_entities": [{"label": "general object label", '
        '"confidence_hint": "low|medium|high", "notes_tr": "short note in Turkish"}],\n'
        '  "detected_actions_tr": ["observed actions, in Turkish"],\n'
        '  "risk_events": [\n'
        '    {\n'
        '      "description_tr": "risk description, in Turkish",\n'
        '      "severity": "low|medium|high|critical",\n'
        '      "confidence": 0.85,\n'
        f'      "timestamp_sec": {clip_start_sec:.1f}\n'
        '    }\n'
        '  ],\n'
        '  "confidence_overall": 0.0\n'
        "}\n"
        "risk_events RULES: return an empty list [] if there is no risk. "
        "severity: low=low priority, medium=needs attention, high=significant risk, "
        "critical=human health in immediate danger. "
        "confidence: your risk-specific confidence (0.0-1.0). "
        "timestamp_sec: the absolute second in the full video when the risk is visible."
    )


class ServerBackend(VLMBackendBase):
    """TEKNOFEST EVREN ve OpenAI-uyumlu harici VLM servisleri adaptörü.

    Hem grid JPEG görsellerini (`model_name`) hem de doğrudan MP4 videoları
    (`video_model`, varsayılan 'vlm') destekler.
    """
    backend_name = "server"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        video_model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 65536,
        temperature: float = 0.15,
        timeout_sec: int = 1800,
        enable_thinking: bool | None = False,
    ) -> None:
        """ServerBackend nesnesini başlatır.

        Args:
            base_url (str): API uç nokta kökü.
            model_name (str): Görsel ve metin modeli (örn. 'llm-large').
            video_model (str | None, optional): Video analizi modeli (örn. 'vlm').
            api_key (str | None, optional): Bearer yetkilendirme anahtarı.
            max_tokens (int, optional): Azami üretim token limiti. Varsayılan: 65536.
            temperature (float, optional): Sıcaklık değeri. Varsayılan: 0.15.
            timeout_sec (int, optional): İstek zaman aşımı (saniye). Varsayılan: 1800.
            enable_thinking (bool | None, optional): Akıl yürütme düşünce izi bayrağı.
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.video_model = video_model or model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self.enable_thinking = enable_thinking

    def _headers(self) -> dict[str, str]:
        """HTTP istek başlıklarını oluşturur."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, model: str, content: list[dict[str, Any]], max_tokens: int | None = None) -> tuple[str, int]:
        """`/chat/completions` uç noktasına HTTP POST isteği gönderir.

        Args:
            model (str): Çağrılacak model adı.
            content (list[dict[str, Any]]): Mesaj içerik blokları.
            max_tokens (int | None, optional): İstek özelinde token limiti.

        Returns:
            tuple[str, int]: (Model yanıt metni, üretilen token adedi).

        Raises:
            RuntimeError: Sunucu 200 harici HTTP yanıtı dönerse veya içerik boşsa.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.temperature,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_sec,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Çıkarım servisi HTTP {resp.status_code}: {resp.text[:400]}"
            )
        data = resp.json()
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        tokens = int(data.get("usage", {}).get("completion_tokens", 0))

        if not text.strip():
            raise RuntimeError(
                f"Model boş içerik döndürdü (finish_reason="
                f"{choice.get('finish_reason')!r}, max_tokens={payload['max_tokens']}). "
                f"Akıl yürütme açıksa token bütçesi izi tarafından tükenmiş olabilir."
            )
        return text, tokens

    def _raw_infer(self, image_path: str, prompt: str) -> tuple[str, int]:
        """Grid JPEG görselini `model_name` modeline gönderir."""
        b64 = _encode_jpeg_b64(image_path)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        return self._post(self.model_name, content)

    @staticmethod
    def encode_video(video_path: str) -> str:
        """Video dosyasını Base64 dizesine dönüştürür ve gövde boyutunu denetler.

        Args:
            video_path (str): Video dosya yolu.

        Returns:
            str: Base64 kodlanmış video verisi.

        Raises:
            FileNotFoundError: Dosya mevcut değilse.
            RuntimeError: Boyut API sınırını aşıyorsa.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video bulunamadı: {video_path}")
        raw = open(video_path, "rb").read()
        if len(raw) > _RAW_VIDEO_LIMIT_BYTES:
            raise RuntimeError(
                f"{os.path.basename(video_path)} {len(raw)/1e6:.1f} MB — base64 "
                f"sonrası gövde sınırını aşar. Klip daha kısa segmentlere bölünmeli."
            )
        return base64.b64encode(raw).decode("utf-8")

    def infer_video(
        self,
        video_path: str,
        video_id: str,
        clip_start_sec: float = 0.0,
        clip_end_sec: float = 0.0,
        memory_context: str = "",
        is_segment: bool = False,
        video_b64: str | None = None,
        packet_id: str | None = None,
    ) -> VLMInterpretation:
        """Video dosyasını doğrudan `video_model` (vlm) modeline göndererek S8 yorumunu üretir.

        Args:
            video_path (str): Analiz edilecek MP4 klibi.
            video_id (str): Video kimliği.
            clip_start_sec (float, optional): Klibin tam videodaki mutlak başlangıç saniyesi.
            clip_end_sec (float, optional): Klibin tam videodaki mutlak bitiş saniyesi.
            memory_context (str, optional): Önceki segmentlerin metin hafızası.
            is_segment (bool, optional): Parçalı video segmenti mi bayrağı.
            video_b64 (str | None, optional): Önceden kodlanmış Base64 verisi (prefix caching için).
            packet_id (str | None, optional): Özel paket UUID'si.

        Returns:
            VLMInterpretation: S8 sözleşmesi nesnesi.
        """
        import uuid

        b64 = video_b64 if video_b64 is not None else self.encode_video(video_path)
        prompt = build_video_system_prompt(
            clip_start_sec, clip_end_sec, memory_context, is_segment
        )
        content = [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
        ]

        t0 = time.perf_counter()
        raw_text, tokens = self._post(self.video_model, content)
        latency_ms = (time.perf_counter() - t0) * 1000

        parse_ok = True
        reasoning = ""
        try:
            reasoning, parsed = _extract_reasoning_and_json(raw_text)
        except ValueError:
            parse_ok = False
            parsed = {
                "scene_summary_tr": "Model çıktısı ayrıştırılamadı.",
                "detected_entities": [],
                "detected_actions_tr": [],
                "risk_events": [],
                "confidence_overall": 0.0,
            }

        entities = [
            DetectedEntity(
                label=e.get("label", ""),
                confidence_hint=e.get("confidence_hint", "low"),
                notes_tr=e.get("notes_tr", ""),
            )
            for e in parsed.get("detected_entities", [])
            if isinstance(e, dict)
        ]

        risk_events = [
            RiskEvent(
                description_tr=r.get("description_tr", ""),
                severity=r.get("severity", "low"),
                confidence=float(r.get("confidence", 0.5)),
                supporting_frame_count=int(r.get("supporting_frame_count", 1)),
                supporting_frame_positions=r.get("supporting_frame_positions", []) or [],
                timestamp_sec=_clamp_timestamp(
                    r.get("timestamp_sec"), clip_start_sec, clip_end_sec
                ),
            )
            for r in parsed.get("risk_events", [])
            if isinstance(r, dict)
        ]

        return VLMInterpretation(
            packet_id=packet_id or str(uuid.uuid4()),
            video_id=video_id,
            model_name=self.video_model,
            model_backend=self.backend_name,
            scene_summary_tr=parsed.get("scene_summary_tr", ""),
            detected_entities=entities,
            detected_actions_tr=parsed.get("detected_actions_tr", []),
            risk_events=risk_events,
            confidence_overall=float(parsed.get("confidence_overall", 0.0) or 0.0),
            computed_confidence=_compute_confidence(parsed, parse_ok),
            inference=InferenceMetrics(latency_ms=latency_ms, tokens_generated=tokens),
            raw_model_output=raw_text,
            reasoning_trace=reasoning,
            segment_start_sec=clip_start_sec,
            segment_end_sec=clip_end_sec,
        )

    def ask_followup(
        self,
        question: str,
        video_b64: str,
        history: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Aynı video üzerinde takip sorusu sorar (ön ek önbelleği / prefix cache isabeti).

        Args:
            question (str): Sorulacak soru metni.
            video_b64 (str): Base64 formatındaki video içeriği.
            history (list[dict[str, Any]] | None, optional): Geçmiş konuşma mesajları.
            max_tokens (int, optional): Yanıt için ayrılan token limiti. Varsayılan: 1024.

        Returns:
            str: Modelin ürettiği metin cevabı.
        """
        messages: list[dict[str, Any]] = list(history) if history else [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Bu videoyu inceleyeceğiz."},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
            ],
        }]
        messages.append({"role": "user", "content": question})

        payload: dict[str, Any] = {
            "model": self.video_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"].get("content") or ""


# ---------------------------------------------------------------------------
# Zaman Damgası Normalizasyonu Fonksiyonu
# ---------------------------------------------------------------------------

def _clamp_timestamp(
    value: object, start_sec: float, end_sec: float
) -> float | None:
    """Modelin ürettiği zaman damgasını klip aralığına oturtur ve mutlak zamana dönüştürür.

    Model bazen segment süresi içinde göreli zaman (örn. 0.0-10.0 sn) veya geçersiz
    değerler üretebilir. Bu fonksiyon:
        - Sayısal olmayan veya None değerleri None yapar.
        - Değer zaten mutlak segment aralığındaysa doğrudan korur.
        - Değer göreli klip süresine uyuyorsa `start_sec + ts` ile mutlak eksene öteler.
        - Aralık dışı sapan değerleri en yakın sınır saniyesine (start_sec / end_sec) çeker.

    Args:
        value (object): Modelin ürettiği `timestamp_sec` verisi.
        start_sec (float): Segmentin mutlak başlangıç saniyesi.
        end_sec (float): Segmentin mutlak bitiş saniyesi.

    Returns:
        float | None: Mutlak saniye değeri veya None.
    """
    if value is None:
        return None
    try:
        ts = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    duration = max(0.0, end_sec - start_sec)

    # Değer zaten mutlak segment aralığındaysa koru
    if end_sec > start_sec and start_sec <= ts <= end_sec:
        return ts

    # Göreli zaman damgası tespiti ve mutlak zamana öteleme
    if start_sec > 0 and 0.0 <= ts <= duration + 1e-6:
        ts = start_sec + ts

    if end_sec > start_sec:
        ts = min(max(ts, start_sec), end_sec)
    return ts
