"""
Kanal B - Veri Sözleşmeleri ve Şemalar Modülü (contracts.py)
============================================================

Bu modül, Kanal B mimarisindeki veri alışverişi sözleşmelerini (data contracts),
ara katman yapılarını ve segmentler arası bağlamsal hafıza nesnelerini tanımlar.

Sözleşme Standartları:
----------------------
1. **S1b Sözleşmesi (:class:`VLMFramePacket`):**
   - Kanal B ön işleme katmanının (:mod:`preprocessing`) çıktısı, VLM çıkarım
     arka yüzünün (:mod:`vlm_backend`) girdisidir.
   - Seçilen video karelerinin kalite metriklerini, grid yerleşimini ve diskteki
     görsel yolunu taşır.

2. **S8 Sözleşmesi (:class:`VLMInterpretation`):**
   - VLM arka yüzünün ürettiği ve Karar Ajanı'na (:mod:`src.reasoning.decision_agent`)
     iletilen nihai görsel yorum paketidir.
   - Sahne özeti, tespit edilen genel varlıklar, gözlemlenen eylemler, derecelendirilmiş
     İSG risk olayları (:class:`RiskEvent`) ve performans metriklerini içerir.

3. **Segmentli / İterasyonlu Sözleşmeler:**
   - :class:`SegmentResult` ve :class:`BatchResult`: Uzun videoların parçalı analiz
     sonuçlarını tek bir potada toplar veya geriye dönük uyumlu tekil S8 yorumuna
     indirger (:meth:`BatchResult.to_interpretation_dict`).
   - :class:`VideoAnalysisMemory`: Segmentler arası bağlamı (varlıklar, kronolojik riskler)
     metin formatında damıtarak bağlam penceresi taşmasını önleyen dinamik hafıza yapısıdır.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal


def _now_iso() -> str:
    """UTC zaman diliminde ISO-8601 standart zaman damgası üretir.

    Returns:
        str: 'YYYY-MM-DDTHH:MM:SS.mmmmmm+00:00' formatında zaman dizesi.
    """
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# S1b — Kanal B Ön İşleme Çıktısı (preprocessing.py → vlm_backend.py)
# ---------------------------------------------------------------------------

@dataclass
class FrameQualityMetrics:
    """Bir kareye ilişkin hesaplanan kalite, keskinlik ve parlaklık metrikleri.

    Bu metrikler hem akıllı kare seçim algoritmasında hem de S1b paketinin denetim
    izinde (audit trail) kullanılır.

    Attributes:
        laplacian_var (float): Laplacian varyansı ile ölçülen keskinlik derecesi.
            Yüksek değerler keskin/net kenarlara, düşük değerler odak/hareket bulanıklığına
            işaret eder.
        ssim_diff (float): Bir önceki seçilmiş/aday kareyle aradaki Yapısal Benzerlik
            (SSIM) farkı (1.0 - SSIM, [0.0 - 1.0] aralığında). Yüksek değer sahne değişimini gösterir.
        brightness_mean (float): Karenin ortalama piksel parlaklığı (0.0=tam siyah, 255.0=tam beyaz).
            CLAHE kontrast iyileştirmesi kararı için eşik değerle kıyaslanır.
    """
    laplacian_var: float
    ssim_diff: float
    brightness_mean: float


@dataclass
class FrameMeta:
    """Seçilen her bir karenin kimlik, zamanlama, yerleşim ve seçim gerekçesi bilgisi.

    Attributes:
        frame_index (int): Orijinal videodaki 0-tabanlı kare numarası.
        timestamp_sec (float): Karenin videonun başlangıcına göre saniye cinsinden mutlak konumu.
        grid_position (int): Kare grid görseline yerleştirildiğindeki hücre sırası (0-indeksli).
        selection_reason (Literal): Karenin neden seçildiğini belirten gerekçe etiketi:
            - "scene_change": SSIM farkı yüksek, belirgin sahne veya açı değişimi.
            - "motion_peak": Optik akış veya hareket tepe noktası.
            - "uniform_sample": Düşük sahne değişimi durumunda periyodik/eşit aralıklı örnekleme.
            - "fallback_fill": Yetersiz aday durumunda zaman ekseninde zorunlu doldurma.
        quality (FrameQualityMetrics): Kareye ait kalite ve parlaklık metrikleri nesnesi.
    """
    frame_index: int
    timestamp_sec: float
    grid_position: int
    selection_reason: Literal[
        "scene_change",
        "motion_peak",
        "uniform_sample",
        "fallback_fill",
    ]
    quality: FrameQualityMetrics


@dataclass
class GridLayout:
    """Oluşturulan grid (mozaik) görselinin geometrik boyut ve ızgara bilgileri.

    VLM modelinin hangi matris düzeninde görüntü aldığını bilmesini sağlar.

    Attributes:
        rows (int): Grid görselindeki satır sayısı (örn. 2).
        cols (int): Grid görselindeki sütun sayısı (örn. 4).
        cell_size (tuple[int, int]): Her bir hücrenin (genişlik, yükseklik) piksel boyutları.
    """
    rows: int
    cols: int
    cell_size: tuple[int, int]


@dataclass
class EnhancementInfo:
    """Görüntü iyileştirme (CLAHE - Kontrast Sınırlı Adaptif Histogram Eşitleme) meta verisi.

    Karar mekanizmasının düşük ışık koşullarında yapay aydınlatma uygulandığını bilmesini sağlar.

    Attributes:
        clahe_applied (bool): Paketteki en az bir kareye CLAHE uygulandıysa True, aksi halde False.
        clip_limit (float): CLAHE algoritmasında uygulanan kontrast sınır katsayısı.
        tile_grid_size (tuple[int, int]): CLAHE algoritmasının mozaik ızgara boyutu (örn. (8, 8)).
    """
    clahe_applied: bool
    clip_limit: float
    tile_grid_size: tuple[int, int]


@dataclass
class VLMFramePacket:
    """S1b Sözleşmesi — Kanal B Ön İşleme Çıktısı ve VLM Arka Yüzünün Girdisi.

    :mod:`preprocessing` modülü tarafından üretilir; :mod:`vlm_backend` modülündeki
    `infer()` fonksiyonuna girdi olarak sağlanır.

    Attributes:
        video_id (str): Analiz edilen videonun benzersiz kimliği.
        source_start_sec (float): Grid paketine dahil edilen ilk karenin zaman damgası (sn).
        source_end_sec (float): Grid paketine dahil edilen son karenin zaman damgası (sn).
        frames (list[FrameMeta]): Seçilen karelerin ayrıntılı meta verileri listesi.
        grid_layout (GridLayout): Grid'in satır, sütun ve hücre boyutlandırma bilgisi.
        enhancement (EnhancementInfo): CLAHE ve ışık iyileştirme bilgileri.
        grid_image_path (str): Diske kaydedilen birleştirilmiş JPEG grid görselinin mutlak yolu.
        packet_id (str): Paket için üretilen benzersiz UUID (S1b ile S8 arasındaki izlenebilirliği sağlar).
        created_at (str): Paketin ISO-8601 formatında UTC oluşturulma zamanı.
    """
    video_id: str
    source_start_sec: float
    source_end_sec: float
    frames: list[FrameMeta]
    grid_layout: GridLayout
    enhancement: EnhancementInfo
    grid_image_path: str
    packet_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        """Paketi JSON serileştirmeye uygun sözlük (dict) yapısına dönüştürür.

        Returns:
            dict: Tüm iç içe dataclass'ları içeren sözlük.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# S8 — VLM Çıktısı (vlm_backend.py → Karar Ajanı)
# ---------------------------------------------------------------------------

@dataclass
class DetectedEntity:
    """VLM tarafından görsel olarak tespit edilen bir nesne veya varlık kaydı.

    Attributes:
        label (str): Varlığın genel tanımı (örn. 'person', 'forklift', 'truck', 'pallet').
            Kanal B genel terimler tercih eder; kesin teknik sınıflandırma algı katmanına aittir.
        confidence_hint (Literal["low", "medium", "high"]): Modelin tespit güven seviyesi.
        notes_tr (str): Tespite dair kısa Türkçe açıklama veya belirsizlik notu.
    """
    label: str
    confidence_hint: Literal["low", "medium", "high"]
    notes_tr: str = ""


@dataclass
class InferenceMetrics:
    """VLM model çağrısına ait çıkarım performansı ve kaynak kullanım metrikleri.

    Attributes:
        latency_ms (float): Model çağrısının ağ ve çıkarım süresi dahil toplam gecikmesi (ms).
        tokens_generated (int): Modelin ürettiği tamamlama (completion) token sayısı.
    """
    latency_ms: float
    tokens_generated: int


@dataclass
class RiskEvent:
    """VLM tarafından görsel kanıtlara dayalı olarak tespit edilen tekil İSG risk durumu.

    Her bir risk olayı; şiddeti, modele özgü güven skoru ve tespit edildiği zaman/ızgara
    kanıtlarıyla birlikte taşınır. Karar Ajanı bu sayede riskleri ağırlıklandırabilir.

    Desteklenen Kanıt Biçimleri:
        - **Grid Modu:** Riskin görüldüğü grid hücre indeksleri `supporting_frame_positions` alanına yazılır.
        - **Video Modu:** Riskin görüldüğü an videonun başından itibaren mutlak saniye olarak
          `timestamp_sec` alanına atanır.

    Attributes:
        description_tr (str): Riskin Türkçe açıklaması (örn. 'Personel baretsiz çalışıyor').
        severity (Literal["low", "medium", "high", "critical"]): Riskin önem derecesi:
            - 'low': Düşük öncelikli durum.
            - 'medium': Dikkat gerektiren kural ihlali.
            - 'high': Ciddi kaza/hasar riski.
            - 'critical': İnsan hayatını doğrudan tehlikeye atan acil durum.
        confidence (float): Riske özgü olasılık/güven skoru (0.0 - 1.0).
        supporting_frame_count (int): Riskin gözlemlendiği kare veya grid hücresi adedi.
        supporting_frame_positions (list[int]): Riskin yer aldığı grid hücre pozisyonları listesi.
        timestamp_sec (float | None): Riskin gerçekleştiği mutlak video saniyesi (video modunda).
    """
    description_tr: str
    severity: Literal["low", "medium", "high", "critical"]
    confidence: float
    supporting_frame_count: int = 1
    supporting_frame_positions: list[int] = field(default_factory=list)
    timestamp_sec: float | None = None


#: Risk şiddetlerinin sayısal ağırlık karşılıkları. Sıralama ve en kritik olayı seçmede kullanılır.
SEVERITY_RANK: dict[str, int] = {"low": 1, "medium": 2, "high": 3, "critical": 4}


@dataclass
class VLMInterpretation:
    """S8 Sözleşmesi — Karar Ajanı'nın 3. Girdisi Olan VLM Çıktısı.

    :mod:`vlm_backend` tarafından üretilir; Karar Ajanı (:mod:`src.reasoning.decision_agent`)
    bu yorumu Kanal A'nın kural motoru çıktılarıyla (S2, S3) füzyon yaparak nihai kararı verir.

    Attributes:
        packet_id (str): İlgili S1b paketiyle izlenebilirlik sağlayan benzersiz UUID.
        video_id (str): Analiz edilen video kimliği.
        model_name (str): Çıkarımda kullanılan modelin adı veya takma adı (örn. 'vlm', 'llm-large').
        model_backend (Literal): Kullanılan çıkarım altyapısı ('vllm', 'llama.cpp', 'transformers', 'server').
        scene_summary_tr (str): 1-3 cümlelik bağımsız Türkçe sahne özeti.
        detected_entities (list[DetectedEntity]): Tespit edilen varlıklar listesi.
        detected_actions_tr (list[str]): Sahnedeki eylemler listesi (Türkçe).
        confidence_overall (float): Modelin kendi beyan ettiği genel güven derecesi (0.0 - 1.0).
        inference (InferenceMetrics): Çıkarım süresi ve token metrikleri.
        raw_model_output (str): Modelden gelen ham metin çıktısı (denetim ve hata ayıklama için).
        risk_events (list[RiskEvent]): Tespit edilen yapılandırılmış risk olayları listesi.
        risk_flags_tr (list[str]): Geriye dönük uyumluluk için düz metin risk açıklamaları listesi.
        computed_confidence (float): Model öz güveninden bağımsız, dışsal metriklerle hesaplanan güven skoru.
        reasoning_trace (str): Modelin akıl yürütme adımları (<think> bloğu içeriği).
        segment_start_sec (float): Yorumun kapsadığı aralığın mutlak başlangıç saniyesi.
        segment_end_sec (float): Yorumun kapsadığı aralığın mutlak bitiş saniyesi.
        created_at (str): Paketin ISO-8601 UTC oluşturulma zamanı.
    """
    packet_id: str
    video_id: str
    model_name: str
    model_backend: Literal["vllm", "llama.cpp", "transformers", "server"]
    scene_summary_tr: str
    detected_entities: list[DetectedEntity]
    detected_actions_tr: list[str]
    confidence_overall: float
    inference: InferenceMetrics
    raw_model_output: str

    # --- Varsayılanı Olan Alanlar ---
    risk_events: list[RiskEvent] = field(default_factory=list)
    risk_flags_tr: list[str] = field(default_factory=list)
    computed_confidence: float = 0.0
    reasoning_trace: str = ""
    segment_start_sec: float = 0.0
    segment_end_sec: float = 0.0
    created_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        """Geriye dönük uyumluluk amacıyla `risk_flags_tr` alanını `risk_events` üzerinden doldurur."""
        if not self.risk_flags_tr and self.risk_events:
            self.risk_flags_tr = [
                ev.description_tr for ev in self.risk_events if ev.description_tr
            ]

    @property
    def max_severity(self) -> str:
        """Yorumdaki en yüksek risk derecesini ('critical', 'high', 'medium', 'low') döndürür."""
        if not self.risk_events:
            return "low"
        return max(self.risk_events, key=lambda e: SEVERITY_RANK.get(e.severity, 0)).severity

    def to_dict(self) -> dict:
        """Yorum nesnesini JSON serileştirilebilir sözlüğe dönüştürür.

        Returns:
            dict: Karar Ajanı'na aktarılan S8 sözlüğü.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Segment Bazlı (İterasyonlu) Analiz Sözleşmeleri
# ---------------------------------------------------------------------------

def format_mmss(seconds: float) -> str:
    """Saniye cinsinden verilen süreyi 'MM:SS' formatına dönüştürür.

    Hem VLM sistem promptlarında hem de Karar Ajanı zaman damgası şemalarında
    tutarlı zaman gösterimi sağlamak amacıyla kullanılır.

    Args:
        seconds (float): Dönüştürülecek saniye miktarı.

    Returns:
        str: '01:23' biçiminde iki basamaklı dakika ve saniye dizesi.
    """
    total = max(0, int(round(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


@dataclass
class SegmentResult:
    """Uzun videolardan kesilen tek bir video segmentinin analiz sonucu.

    Attributes:
        segment_index (int): Segmentin 0-indeksli sıra numarası.
        start_sec (float): Segmentin ana videodaki mutlak başlangıç saniyesi.
        end_sec (float): Segmentin ana videodaki mutlak bitiş saniyesi.
        interpretation (VLMInterpretation): Segment için üretilen bağımsız S8 yorumu.
    """
    segment_index: int
    start_sec: float
    end_sec: float
    interpretation: VLMInterpretation

    @property
    def duration_sec(self) -> float:
        """Segmentin saniye cinsinden süresi."""
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def max_severity(self) -> str:
        """Segment içinde tespit edilen en yüksek risk şiddeti derecesi."""
        return self.interpretation.max_severity

    def to_dict(self) -> dict:
        """Segment sonucunu sözlüğe serileştirir.

        Returns:
            dict: Zaman aralığı, şiddet ve yorum detaylarını içeren sözlük.
        """
        return {
            "segment_index": self.segment_index,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "time_range": f"{format_mmss(self.start_sec)}-{format_mmss(self.end_sec)}",
            "max_severity": self.max_severity,
            "interpretation": self.interpretation.to_dict(),
        }


@dataclass
class BatchResult:
    """Parçalı (segmentli) analiz edilen bir videonun toplu sonuç kümesi.

    Attributes:
        video_id (str): Analiz edilen videonun kimliği.
        segments (list[SegmentResult]): Başarıyla işlenen segment sonuçları listesi.
        failed_segments (list[dict]): İşlenirken hata alan segmentlerin hata raporları listesi.
        memory_context (str): Segmentler arası taşınan metin hafızasının son durumu.
        created_at (str): ISO-8601 UTC oluşturulma zamanı.
    """
    video_id: str
    segments: list[SegmentResult] = field(default_factory=list)
    failed_segments: list[dict] = field(default_factory=list)
    memory_context: str = ""
    created_at: str = field(default_factory=_now_iso)

    @property
    def total_segments(self) -> int:
        """Başarıyla analiz edilmiş toplam segment adedi."""
        return len(self.segments)

    @property
    def total_risk_events(self) -> int:
        """Tüm segmentlerdeki risk olaylarının toplam adedi."""
        return sum(len(s.interpretation.risk_events) for s in self.segments)

    @property
    def most_critical_segment(self) -> SegmentResult | None:
        """En yüksek şiddette risk içeren segment nesnesi; risk yoksa None."""
        scored = [s for s in self.segments if s.interpretation.risk_events]
        if not scored:
            return None
        return max(
            scored,
            key=lambda s: (
                SEVERITY_RANK.get(s.max_severity, 0),
                max((e.confidence for e in s.interpretation.risk_events), default=0.0),
            ),
        )

    @property
    def all_risk_events(self) -> list[RiskEvent]:
        """Tüm segmentlerin risk olaylarının kronolojik sırada birleştirilmiş listesi."""
        events: list[RiskEvent] = []
        for seg in sorted(self.segments, key=lambda s: s.start_sec):
            events.extend(seg.interpretation.risk_events)
        return events

    def to_dict(self) -> dict:
        """Toplu sonucu JSON formatına uygun sözlüğe çevirir."""
        critical = self.most_critical_segment
        return {
            "video_id": self.video_id,
            "total_segments": self.total_segments,
            "total_risk_events": self.total_risk_events,
            "most_critical_segment_index": critical.segment_index if critical else None,
            "memory_context": self.memory_context,
            "segments": [s.to_dict() for s in self.segments],
            "failed_segments": self.failed_segments,
            "created_at": self.created_at,
        }

    def to_interpretation_dict(self) -> dict:
        """Segment sonuçlarını tek bir S8 yorum sözlüğüne indirger (aggregate eder).

        Geriye dönük uyumluluk sağlamak adına :mod:`src.reasoning.decision_agent` gibi
        tekil S8 sözlüğü bekleyen tüketicilere birleştirilmiş çıktı sunar.
        Segment özetleri zaman aralıklarıyla etiketlenir, tespit edilen varlıklar
        tekilleştirilir ve riskler kronolojik sırayla toplanır.

        Returns:
            dict: Tek bir S8 VLMInterpretation nesnesinin `to_dict()` yapısıyla uyumlu sözlük.
        """
        if not self.segments:
            return {
                "scene_summary_tr": "Video analiz edilemedi.",
                "detected_entities": [],
                "detected_actions_tr": [],
                "risk_events": [],
                "risk_flags_tr": [],
                "confidence_overall": 0.0,
                "computed_confidence": 0.0,
                "segment_count": 0,
                "failed_segments": self.failed_segments,
            }

        ordered = sorted(self.segments, key=lambda s: s.start_sec)

        summary_parts: list[str] = []
        actions: list[str] = []
        entities: list[dict] = []
        seen_entities: set[str] = set()
        risk_events: list[dict] = []

        for seg in ordered:
            interp = seg.interpretation
            if interp.scene_summary_tr:
                label = f"[{format_mmss(seg.start_sec)}-{format_mmss(seg.end_sec)}]"
                summary_parts.append(f"{label} {interp.scene_summary_tr}")
            actions.extend(interp.detected_actions_tr)
            for ent in interp.detected_entities:
                if ent.label and ent.label.lower() not in seen_entities:
                    seen_entities.add(ent.label.lower())
                    entities.append(asdict(ent))
            for ev in interp.risk_events:
                item = asdict(ev)
                if item.get("timestamp_sec") is None:
                    item["timestamp_sec"] = seg.start_sec
                item["timestamp"] = format_mmss(float(item["timestamp_sec"]))
                risk_events.append(item)

        risk_events.sort(key=lambda e: e.get("timestamp_sec") or 0.0)
        n = len(ordered)

        return {
            "scene_summary_tr": " ".join(summary_parts),
            "detected_entities": entities,
            "detected_actions_tr": actions,
            "risk_events": risk_events,
            "risk_flags_tr": [e["description_tr"] for e in risk_events if e.get("description_tr")],
            "confidence_overall": sum(s.interpretation.confidence_overall for s in ordered) / n,
            "computed_confidence": sum(s.interpretation.computed_confidence for s in ordered) / n,
            "segment_count": n,
            "failed_segments": self.failed_segments,
            "memory_context": self.memory_context,
        }


# ---------------------------------------------------------------------------
# Metin Tabanlı Anlamsal ve Zamansal Hafıza (İterasyonlu Analiz İçin)
# ---------------------------------------------------------------------------

@dataclass
class VideoAnalysisMemory:
    """Segmentler arasında taşınan, metin tabanlı anlamsal ve zamansal hafıza yöneticisi.

    Tasarım Mantığı:
        Uzun videolarda önceki segmentlerin ham yanıtlarını veya token dizilerini
        biriktirmek LLM bağlam penceresini (context window) hızla tüketir.
        Bu sınıf, her segmentin S8 yorumundan yalnızca öz anlamsal bilgileri (varlıklar,
        mekan özellikleri) ve zaman damgalı risk olaylarını damıtır. Belirlenen
        tavan limitlere (`max_semantic_notes`, `max_temporal_events`) ulaşıldığında
        en eski kayıtları budayarak sabit bir bağlam bütçesi sağlar.

    Attributes:
        semantic_notes (list[str]): Mekan ve varlık niteliklerini tutan notlar.
        temporal_events (list[str]): Zaman damgalı olay ve risk satırları.
        max_semantic_notes (int): Tutulacak azami anlamsal not adedi (varsayılan 12).
        max_temporal_events (int): Tutulacak azami zamansal olay adedi (varsayılan 40).
        max_note_chars (int): Bir anlamsal notun azami karakter uzunluğu (varsayılan 240).
    """
    semantic_notes: list[str] = field(default_factory=list)
    temporal_events: list[str] = field(default_factory=list)
    max_semantic_notes: int = 12
    max_temporal_events: int = 40
    max_note_chars: int = 240

    def absorb(self, interpretation: VLMInterpretation, segment_start_sec: float = 0.0) -> None:
        """Yeni bir segment yorumunu işleyerek hafızaya damıtır ve ekler.

        Args:
            interpretation (VLMInterpretation): Segmentten elde edilen S8 yorum nesnesi.
            segment_start_sec (float, optional): Segmentin ana videodaki mutlak başlangıç saniyesi.
        """
        summary = (interpretation.scene_summary_tr or "").strip()
        if summary:
            note = f"[{format_mmss(segment_start_sec)}] {summary[: self.max_note_chars]}"
            if note not in self.semantic_notes:
                self.semantic_notes.append(note)

        labels = [e.label for e in interpretation.detected_entities if e.label]
        if labels:
            uniq = sorted(set(labels))
            note = f"[{format_mmss(segment_start_sec)}] görülen varlıklar: {', '.join(uniq)}"
            if note not in self.semantic_notes:
                self.semantic_notes.append(note)

        for ev in interpretation.risk_events:
            when = ev.timestamp_sec if ev.timestamp_sec is not None else segment_start_sec
            line = (
                f"{format_mmss(float(when))} — {ev.description_tr} "
                f"[{ev.severity}, güven {ev.confidence:.2f}]"
            )
            if line not in self.temporal_events:
                self.temporal_events.append(line)

        self._trim()

    def _trim(self) -> None:
        """Hafıza boyutlarını tanımlı azami sınırlara kırpar (en güncel kayıtları korur)."""
        if len(self.semantic_notes) > self.max_semantic_notes:
            self.semantic_notes = self.semantic_notes[-self.max_semantic_notes :]
        if len(self.temporal_events) > self.max_temporal_events:
            self.temporal_events = self.temporal_events[-self.max_temporal_events :]

    def is_empty(self) -> bool:
        """Hafızada kayıt bulunup bulunmadığını kontrol eder.

        Returns:
            bool: Anlamsal veya zamansal hiçbir kayıt yoksa True.
        """
        return not self.semantic_notes and not self.temporal_events

    def to_prompt_context(self) -> str:
        """Hafızadaki bilgileri bir sonraki segmentin VLM sistem promptuna eklenecek metne çevirir.

        Returns:
            str: Prompt içerisine enjekte edilmeye hazır özet bağlam metni.
        """
        if self.is_empty():
            return ""
        lines = ["ÖNCEKİ SEGMENTLERDEN TAŞINAN BAĞLAM (metin hafıza):"]
        if self.semantic_notes:
            lines.append("Sahne ve varlıklar:")
            lines.extend(f"  - {n}" for n in self.semantic_notes)
        if self.temporal_events:
            lines.append("Zaman damgalı olaylar:")
            lines.extend(f"  - {e}" for e in self.temporal_events)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Hafıza durumunu sözlüğe dönüştürür."""
        return {
            "semantic_notes": list(self.semantic_notes),
            "temporal_events": list(self.temporal_events),
        }
