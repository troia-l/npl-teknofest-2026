# Kanal B Geliştirici & Mimari Notları

Kanal B detaylı dokümantasyonu için lütfen [README.md](README.md) dosyasına bakınız.

### Önemli Noktalar:
1. **Modül İsimlendirmesi:** Depo kökünde mikroservisleri barındıran bir `backend/` paketi bulunduğu için bu klasördeki VLM motoru `vlm_backend.py` adını taşır (`backend.py` DEĞİLDİR).
2. **Zaman Damgaları:** TEKNOFEST EVREN servisine gönderilen segmentler için prompt mutlak zaman damgası (`clip_start_sec`) ile beslenir; `_clamp_timestamp` fonksiyonu model çıktısındaki zaman damgalarını doğrular ve sınırlandırır.
3. **Ön Ek Önbelleği (Prefix Caching):** 60 saniyeden kısa ve 720p altındaki videolarda dosya yeniden kodlanmadan doğrudan iletilir.
