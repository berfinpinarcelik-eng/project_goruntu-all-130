# =============================================================================
# 🔍 AI Görsel Tanıma Uygulaması - Groq Cloud API ile
# Geliştirici: Nejdet TUT - 2026
# Açıklama: Bu uygulama, Groq'un multimodal LLM modellerini kullanarak
#           yüklenen herhangi bir görseldeki nesneleri tanımlar ve analiz eder.
# =============================================================================

import streamlit as st
import base64
import os
import json
import re
from io import BytesIO
from PIL import Image
from groq import Groq
from dotenv import load_dotenv

# .env dosyasından ortam değişkenlerini yükle
load_dotenv()

# ========================= SAYFA YAPILANDIRMASI ==============================
st.set_page_config(
    page_title="AI Görsel Tanıma | Groq Vision",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= ÖZEL CSS STİLLERİ =================================
# Premium ve modern bir görünüm için kapsamlı CSS tanımları
st.markdown("""
<style>
    /* ---------- Google Fonts ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* ---------- Genel Sayfa Stili ---------- */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Sidebar Stili ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #131336 0%, #1c1c4a 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #c4b5fd !important;
        font-size: 0.92rem;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0d4fc !important;
    }

    /* ---------- Başlık Stili ---------- */
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 30%, #ec4899 70%, #f97316 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        color: #a5b4fc;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 2rem;
        opacity: 0.85;
    }

    /* ---------- Yükleme Alanı ---------- */
    .stFileUploader > div > div {
        background: rgba(139, 92, 246, 0.08) !important;
        border: 2px dashed rgba(139, 92, 246, 0.35) !important;
        border-radius: 16px !important;
        transition: all 0.3s ease;
    }
    .stFileUploader > div > div:hover {
        border-color: rgba(139, 92, 246, 0.7) !important;
        background: rgba(139, 92, 246, 0.12) !important;
    }

    /* ---------- Sonuç Kartları ---------- */
    .result-card {
        background: linear-gradient(145deg, rgba(30, 27, 75, 0.95), rgba(49, 46, 129, 0.7));
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 20px;
        padding: 28px 32px;
        margin: 16px 0;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }

    /* ---------- Nesne İsim Etiketi ---------- */
    .object-name {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #e0d4fc;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* ---------- Açıklama Metni ---------- */
    .description-text {
        color: #c4b5fd;
        font-size: 1.05rem;
        line-height: 1.7;
        margin: 12px 0;
        padding: 16px 20px;
        background: rgba(139, 92, 246, 0.08);
        border-radius: 12px;
        border-left: 4px solid #7c3aed;
    }

    /* ---------- Güven Skoru Barı ---------- */
    .confidence-container {
        margin-top: 20px;
        padding: 16px 20px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 12px;
    }
    .confidence-label {
        color: #a5b4fc;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .confidence-bar-bg {
        width: 100%;
        height: 14px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 7px;
        overflow: hidden;
        position: relative;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 7px;
        transition: width 1s ease-in-out;
        position: relative;
    }
    .confidence-bar-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 2s infinite;
    }
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    .confidence-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        text-align: right;
        margin-top: 8px;
    }

    /* ---------- Durum Rozeti ---------- */
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-error {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* ---------- Bilgi Kartı (Sidebar) ---------- */
    .info-card {
        background: linear-gradient(145deg, rgba(49, 46, 129, 0.5), rgba(30, 27, 75, 0.8));
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 14px;
        padding: 18px;
        margin: 12px 0;
    }

    /* ---------- Görsel Çerçevesi ---------- */
    .image-frame {
        border-radius: 16px;
        overflow: hidden;
        border: 2px solid rgba(139, 92, 246, 0.2);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .image-frame img {
        border-radius: 14px;
    }

    /* ---------- Animasyon ---------- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeInUp 0.6s ease-out forwards;
    }

    /* ---------- Footer Stili ---------- */
    .footer {
        text-align: center;
        color: #6366f1;
        font-size: 0.85rem;
        padding: 24px 0 12px;
        opacity: 0.6;
        border-top: 1px solid rgba(99, 102, 241, 0.15);
        margin-top: 3rem;
    }

    /* ---------- Spinner Stili ---------- */
    .stSpinner > div {
        border-color: #7c3aed !important;
    }

    /* ---------- Genel Metin Renkleri ---------- */
    .stMarkdown p { color: #c4b5fd; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #e0d4fc; }
</style>
""", unsafe_allow_html=True)




# ========================= YARDIMCI FONKSİYONLAR ============================

def gorsel_to_base64(gorsel: Image.Image) -> str:
    """
    PIL Image nesnesini Base64 string'e dönüştürür.
    Groq API'ye görsel göndermek için bu format gereklidir.
    """
    # Görseli byte buffer'a kaydet
    buffer = BytesIO()
    gorsel.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    # Byte verisini Base64'e çevir
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_str


def gorseli_analiz_et(client: Groq, base64_gorsel: str, model_adi: str) -> dict:
    """
    Groq Vision API'ye görseli göndererek analiz sonucunu alır.
    
    Parametreler:
        client: Groq API istemcisi
        base64_gorsel: Base64 formatında görsel verisi
        model_adi: Kullanılacak Groq model adı
    
    Döndürür:
        dict: {"nesne": str, "aciklama": str, "guven_skoru": float}
    """
    # ---- KULLANICI MESAJI (Talimatlar + Görsel birlikte) ----
    # Groq Llama 4 Scout API'si için talimatlar ve görsel aynı user mesajında gönderilir
    kullanici_mesaji = [
        {
            "type": "text",
            "text": """Sen uzman bir görüntü analiz yapay zekasısın. Bu görseli detaylı analiz et ve içindeki ana nesneyi tanımla.

Görevlerin:
1. Görseldeki ANA nesneyi veya sahneyi kesin ve net bir biçimde tespit et.
2. Tespit ettiğin nesne hakkında kısa ama bilgilendirici bir açıklama yaz (2-3 cümle).
3. Tahminin için 0 ile 100 arasında bir güven skoru (confidence score) belirle.

ÖNEMLİ:
- Lütfen ne gördüğünden TAM OLARAK emin değilsen uydurma. "Kırmızı kedi" gibi şeyler uydurma.
- Emin değilsen 'nesne' alanına "Belirsiz" yaz.

YANIT FORMATINI KESİNLİKLE aşağıdaki JSON yapısında ver, başka hiçbir şey ekleme:
{
    "nesne": "Tespit edilen nesnenin Türkçe adı",
    "aciklama": "Nesne hakkında kısa bilgilendirici açıklama",
    "guven_skoru": 85
}"""
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_gorsel}"
            }
        }
    ]

    # ---- GROQ API ÇAĞRISI ----
    try:
        yanit = client.chat.completions.create(
            model=model_adi,
            messages=[
                {"role": "user", "content": kullanici_mesaji}
            ],
            temperature=0.1,       # Uydurmayı önlemek için minimum sıcaklık
            max_completion_tokens=1024,
            response_format={"type": "json_object"}, # Groq JSON Modu Açık
        )

        # API yanıtından metin içeriğini al
        yanit_metni = yanit.choices[0].message.content.strip()

        # JSON yanıtındaki olası markdown kod bloğunu temizle
        yanit_metni = re.sub(r'```json\s*', '', yanit_metni)
        yanit_metni = re.sub(r'```\s*', '', yanit_metni)
        yanit_metni = yanit_metni.strip()

        # JSON'u Python sözlüğüne dönüştür
        sonuc = json.loads(yanit_metni)

        # Güven skorunun geçerli aralıkta olduğundan emin ol
        sonuc["guven_skoru"] = max(0, min(100, int(sonuc.get("guven_skoru", 50))))

        return sonuc

    except json.JSONDecodeError:
        # API JSON döndürmediyse, ham metni kullan
        return {
            "nesne": "Bilinmeyen Nesne",
            "aciklama": yanit_metni if 'yanit_metni' in dir() else "Analiz tamamlanamadı.",
            "guven_skoru": 50
        }
    except Exception as e:
        # Diğer hataları yakala
        return {
            "nesne": "Hata",
            "aciklama": f"API isteği sırasında bir hata oluştu: {str(e)}",
            "guven_skoru": 0
        }


def guven_skoru_rengi(skor: int) -> tuple:
    """
    Güven skoruna göre gradient renkleri ve rozet sınıfı döndürür.
    Yüksek skor = Yeşil, Orta = Turuncu, Düşük = Kırmızı
    """
    if skor >= 80:
        return ("#10b981", "#34d399", "badge-success", "Yüksek Güven")
    elif skor >= 50:
        return ("#f59e0b", "#fbbf24", "badge-warning", "Orta Güven")
    else:
        return ("#ef4444", "#f87171", "badge-error", "Düşük Güven")


# ========================= SIDEBAR (SOL PANEL) ===============================
with st.sidebar:
    # Uygulama logosu ve başlığı
    st.markdown("""
    <div style='text-align: center; padding: 20px 0 10px;'>
        <span style='font-size: 3rem;'>🔍</span>
        <h2 style='font-family: Outfit, sans-serif; background: linear-gradient(135deg, #a78bfa, #ec4899); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-top: 8px;'>
        AI Görsel Tanıma</h2>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("---")

    # ---- Uygulama Hakkında Bilgi Kartı ----
    st.markdown("""
    <div class='info-card'>
        <h4 style='color: #e0d4fc; margin-top: 0;'>📋 Uygulama Hakkında</h4>
        <p style='font-size: 0.88rem; line-height: 1.6;'>
            Bu uygulama <strong style='color: #a78bfa;'>Groq Cloud API</strong> ve 
            <strong style='color: #a78bfa;'>LLaMA Vision</strong> modelini kullanarak
            yüklediğiniz görsellerdeki nesneleri yapay zeka ile tanımlar.
        </p>
        <ul style='font-size: 0.85rem; padding-left: 18px;'>
            <li>🖼️ Her türlü nesneyi tanır</li>
            <li>🎯 Güven skoru ile sonuç verir</li>
            <li>⚡ Saniyeler içinde analiz eder</li>
            <li>🌍 Türkçe açıklama üretir</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("---")

    # ---- API Anahtarı Kontrolü ----
    st.markdown("### 🔑 API Durumu")

    # API anahtarını önce ortam değişkenlerinden, sonra Streamlit secrets'dan al
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

    if api_key:
        # API anahtarı bulundu - yeşil rozet göster
        st.markdown("""
        <div class='info-card' style='border-color: rgba(16, 185, 129, 0.3);'>
            <span class='status-badge badge-success'>✅ API Bağlı</span>
            <p style='font-size: 0.82rem; margin-top: 10px;'>
                Groq API anahtarı başarıyla yüklendi. Uygulama kullanıma hazır.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # API anahtarı bulunamadı - uyarı göster
        st.markdown("""
        <div class='info-card' style='border-color: rgba(239, 68, 68, 0.3);'>
            <span class='status-badge badge-error'>❌ API Bulunamadı</span>
            <p style='font-size: 0.82rem; margin-top: 10px;'>
                <code>.env</code> dosyasına veya Streamlit Secrets'a 
                <code>GROQ_API_KEY</code> ekleyin.
            </p>
        </div>
        """, unsafe_allow_html=True)


        # Manuel giriş alanı (geliştirme amaçlı)
        api_key = st.text_input(
            "API Anahtarını Girin:",
            type="password",
            placeholder="gsk_...",
            help="groq.com adresinden ücretsiz API anahtarı alabilirsiniz."
        )

    st.markdown("---")

    # ---- Model Seçimi ----
    st.markdown("### 🤖 Model Seçimi")
    secilen_model = st.selectbox(
        "Groq Görsel Analiz Modeli (Mart 2026):",
        options=[
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ],
        index=0,
        help="Llama 4 Scout: Mart 2026 itibarıyla fotoğrafları işleyebilen (Multimodal) en güncel resmi Groq modelidir."
    )

    st.markdown("---")

    # ---- Teknoloji Bilgisi ----
    st.markdown("""
    <div class='info-card'>
        <h4 style='color: #e0d4fc; margin-top: 0;'>🛠️ Teknolojiler</h4>
        <p style='font-size: 0.82rem; line-height: 1.8;'>
            🐍 Python & Streamlit<br>
            🤖 Groq Cloud API<br>
            🧠 LLaMA 3.2 Vision<br>
            🖼️ Pillow (PIL)
        </p>
    </div>
    """, unsafe_allow_html=True)



# ========================= ANA EKRAN ========================================

# ---- Ana Başlık ----
st.markdown('<h1 class="main-title">🔍 AI Görsel Tanıma</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Herhangi bir fotoğraf yükleyin — yapay zeka ne olduğunu söylesin</p>',
    unsafe_allow_html=True
)

# ---- İki Sütunlu Düzen ----
col_sol, col_sag = st.columns([1, 1], gap="large")

with col_sol:
    st.markdown("### 📤 Görsel Yükleme")
    # Dosya yükleme bileşeni (JPG, PNG, JPEG, WEBP destekli)
    yuklenen_dosya = st.file_uploader(
        "Bir fotoğraf seçin veya sürükleyip bırakın",
        type=["jpg", "jpeg", "png", "webp"],
        help="Desteklenen formatlar: JPG, JPEG, PNG, WEBP",
        label_visibility="collapsed"
    )

    if yuklenen_dosya is not None:
        # Yüklenen görseli aç ve göster
        gorsel = Image.open(yuklenen_dosya).convert("RGB")
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(gorsel, caption="📸 Yüklenen Görsel", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

        # Dosya bilgileri
        dosya_boyutu = yuklenen_dosya.size / 1024  # KB cinsinden
        st.markdown(f"""
        <div style='display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap;'>
            <span class='status-badge badge-success'>📐 {gorsel.width}×{gorsel.height} px</span>
            <span class='status-badge badge-success'>📦 {dosya_boyutu:.1f} KB</span>
            <span class='status-badge badge-success'>🖼️ {yuklenen_dosya.type.split('/')[-1].upper()}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Yükleme bekleniyor - bilgilendirme mesajı
        st.markdown("""
        <div class='result-card' style='text-align: center; padding: 60px 30px;'>
            <span style='font-size: 4rem; display: block; margin-bottom: 16px;'>📸</span>
            <p style='color: #a5b4fc; font-size: 1.1rem;'>
                Analiz etmek istediğiniz bir fotoğraf yükleyin
            </p>
            <p style='color: #6366f1; font-size: 0.85rem; margin-top: 8px;'>
                JPG, PNG veya WEBP formatında
            </p>
        </div>
        """, unsafe_allow_html=True)

with col_sag:
    st.markdown("### 🧠 AI Analiz Sonucu")

    if yuklenen_dosya is not None and api_key:
        # ---- Analiz Başlat ----
        with st.spinner("🔍 Yapay zeka görseli analiz ediyor..."):
            # Groq API istemcisini oluştur
            client = Groq(api_key=api_key)

            # Görseli Base64'e dönüştür
            base64_veri = gorsel_to_base64(gorsel)

            # API'ye gönder ve sonucu al
            sonuc = gorseli_analiz_et(client, base64_veri, secilen_model)

        # ---- Sonuçları Göster ----
        nesne_adi = sonuc.get("nesne", "Bilinmeyen")
        aciklama = sonuc.get("aciklama", "Açıklama bulunamadı.")
        guven = sonuc.get("guven_skoru", 0)

        # Güven skoruna göre renk belirle
        renk_koyu, renk_acik, rozet_sinifi, guven_etiketi = guven_skoru_rengi(guven)

        # ---- Sonuç Kartı ----
        sonuc_html = f"""
<div class='result-card animate-in'>
    <!-- Nesne Adı -->
    <div class='object-name'>
        <span>🎯</span>
        <span>{nesne_adi}</span>
    </div>
    
    <!-- Durum Rozeti -->
    <span class='status-badge {rozet_sinifi}'>{guven_etiketi}</span>
    
    <!-- Açıklama -->
    <div class='description-text'>
        {aciklama}
    </div>
    
    <!-- Güven Skoru Barı -->
    <div class='confidence-container'>
        <div class='confidence-label'>📊 Güven Skoru</div>
        <div class='confidence-bar-bg'>
            <div class='confidence-bar-fill' 
                 style='width: {guven}%; background: linear-gradient(90deg, {renk_koyu}, {renk_acik});'>
            </div>
        </div>
        <div class='confidence-value' style='color: {renk_acik};'>
            %{guven}
        </div>
    </div>
</div>
"""
        st.markdown(sonuc_html.replace('\n', ' '), unsafe_allow_html=True)

        # ---- Ham JSON Verisi (Genişletilebilir) ----
        with st.expander("📋 Ham API Yanıtı (JSON)", expanded=False):
            st.json(sonuc)

    elif yuklenen_dosya is not None and not api_key:
        # API anahtarı girilmemiş uyarısı
        st.markdown("""
        <div class='result-card' style='text-align: center; border-color: rgba(239, 68, 68, 0.3);'>
            <span style='font-size: 3rem; display: block; margin-bottom: 12px;'>🔐</span>
            <p style='color: #f87171; font-size: 1.1rem; font-weight: 600;'>
                API Anahtarı Gerekli
            </p>
            <p style='color: #a5b4fc; font-size: 0.9rem; margin-top: 8px;'>
                Sol panelden Groq API anahtarınızı girin veya<br>
                <code>.env</code> dosyasına <code>GROQ_API_KEY</code> ekleyin.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Henüz görsel yüklenmemiş
        st.markdown("""
        <div class='result-card' style='text-align: center; padding: 60px 30px;'>
            <span style='font-size: 4rem; display: block; margin-bottom: 16px;'>🤖</span>
            <p style='color: #a5b4fc; font-size: 1.1rem;'>
                Analiz sonuçları burada görüntülenecek
            </p>
            <p style='color: #6366f1; font-size: 0.85rem; margin-top: 8px;'>
                Sol tarafa bir fotoğraf yükleyerek başlayın
            </p>
        </div>
        """, unsafe_allow_html=True)



# ========================= ALT BİLGİ (FOOTER) ===============================
st.markdown("""
<div class='footer'>
    <p>Nejdet TUT — AI Görsel Tanıma Projesi — 2026</p>
    <p style='font-size: 0.75rem; margin-top: 4px;'>Powered by Groq Cloud & LLaMA Vision</p>
</div>
""", unsafe_allow_html=True)

