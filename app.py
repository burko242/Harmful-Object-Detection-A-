import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(page_title="YOLO Güvenlik AI", layout="wide")
st.title("YOLO26 İle Zararlı Nesne Sınıflandırma ve Tespit Sistemi")
st.markdown("Lütfen analiz edilecek bir görsel yükleyin.")

uploaded_file = st.file_uploader("Görsel Seçin (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Orijinal Görsel")
        orijinal_resim = Image.open(uploaded_file)
        st.image(orijinal_resim, width="stretch")
    
    if st.button("Güvenlik Taramasını Başlat"):
        with st.spinner("Yapay Zeka Görseli Sınıflandırıyor..."):
            
            # Hata veren sondaki "/" işareti kaldırıldı
            api_url = "http://127.0.0.1:8000/predict"
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                # Modelin yüklenmesi için 60 saniye süre (timeout) tanındı
                response = requests.post(api_url, files=files, timeout=60)
                
                if response.status_code == 200:
                    veri = response.json()
                    
                    with col2:
                        st.subheader("Yapay Zeka Analiz Sonucu")
                        
                        if "isaretli_resim_base64" in veri:
                            base64_kodu = veri["isaretli_resim_base64"]
                            resim_bytes = base64.b64decode(base64_kodu)
                            isaretli_resim = Image.open(io.BytesIO(resim_bytes))
                            st.image(isaretli_resim, width="stretch")
                        else:
                            st.image(orijinal_resim, width="stretch")
                        
                        if len(veri["tespitler"]) > 0:
                            st.error(f"DİKKAT! {len(veri['tespitler'])} adet potansiyel tehlike sınıflandırıldı.")
                            
                            for tespit in veri["tespitler"]:
                                st.write(f"- **{tespit['nesne']}** (Eminlik: %{tespit['guven']})")
                        else:
                            st.success("Görsel temiz. Herhangi bir zararlı sınıf bulunamadı.")
                            
                else:
                    st.error("API sunucusundan bir hata döndü.")
                    
            except requests.exceptions.ConnectionError:
                st.error("API'ye ulaşılamıyor!")
