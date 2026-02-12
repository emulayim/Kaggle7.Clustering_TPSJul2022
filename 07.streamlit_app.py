import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="TPS Jul 2022 Clustering",
    page_icon="🧩",
    layout="wide"
)

# --- Helper Functions ---
def resolve_model_path(filename):
    """
    Model dosyasını scriptin bulunduğu konuma göre dinamik olarak arar.
    """
    # 1. Şu an çalışan dosyanın (streamlit_app.py) tam yolunu bul
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        # SENARYO 1: Hugging Face (Veya aynı klasörde)
        # Dosya script ile yan yanaysa (örn: src/best_model.pkl)
        os.path.join(current_script_dir, filename),
        
        # SENARYO 2: Lokal Çalışma
        # Script 'src' içinde, model bir üstteki 'models' klasöründe ise
        # (örn: .../Proje/src/.. -> .../Proje/models/best_model.pkl)
        os.path.join(current_script_dir, "..", "models", filename),
        
        # SENARYO 3: Çalışma Dizini (Fallback)
        # Terminalin açıldığı yerdeki 'models' klasörü
        os.path.join("models", filename),
        
        # SENARYO 4: Direkt dosya ismi
        filename
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    # Hata ayıklama için (Opsiyonel: Streamlit arayüzüne basılabilir)
    print(f"Aranan yollar: {possible_paths}")
    return None

@st.cache_resource
def load_model():
    # Model ismi
    model_filename = "best_model.pkl"
    model_path = resolve_model_path(model_filename)
    
    if model_path:
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Model yüklenirken hata oluştu: {e}")
            return None
    return None

# --- Main App ---
def main():
    st.title("🧩 Tabular Clustering (TPS Jul 2022)")
    
    model = load_model()

    if model is None:
        st.error("🚨 Model dosyası (`best_model.pkl`) bulunamadı! Lütfen notebook'u çalıştırıp modeli eğitin veya dosya konumunu kontrol edin.")
        # Hata ayıklama ipucu
        st.info("İpucu: Eğer lokalde 'src' klasöründen çalıştırıyorsanız modelin '../models/' altında olduğundan emin olun.")
        return

    tab1, tab2 = st.tabs(["📝 Manual Input", "📁 Batch Prediction (CSV)"])

    # --- TAB 1: Manual Input ---
    with tab1:
        st.subheader("Predict Cluster for Single Data Point")
        st.info("Enter values for features f_00 to f_28. Default is 0.0.")
        
        # Create input fields dynamically
        input_data = {}
        cols = st.columns(5) # 5 columns layout
        for i in range(29):
            feat_name = f"f_{i:02d}"
            with cols[i % 5]:
                input_data[feat_name] = st.number_input(feat_name, value=0.0)
        
        if st.button("Predict Cluster"):
            df_input = pd.DataFrame([input_data])
            try:
                cluster = model.predict(df_input)[0]
                
                st.divider()
                st.metric("Predicted Cluster", str(cluster))
                
                # Clustering modellerinde predict_proba her zaman olmaz (örn: K-Means)
                # Varsa gösterir, yoksa pas geçer
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(df_input)[0]
                    # Chart
                    chart_df = pd.DataFrame({"Cluster": range(len(proba)), "Probability": proba})
                    fig = px.bar(chart_df, x="Cluster", y="Probability", title="Cluster Probabilities")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Bu model olasılık değerleri (probability) döndürmüyor.")
                    
            except Exception as e:
                st.error(f"Tahmin Hatası: {e}")

    # --- TAB 2: Batch Input ---
    with tab2:
        st.subheader("Batch Clustering")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                # Check for feature columns f_00 to f_28
                required_cols = [f"f_{i:02d}" for i in range(29)]
                missing = [c for c in required_cols if c not in df.columns]
                
                if missing:
                    st.warning(f"Eksik kolonlar var: {missing}. Pipeline bu kolonları bekliyorsa hata alabilirsiniz.")
                
                if st.button("Run Clustering"):
                    # Select only features if ID exists
                    # Eğer eksik kolon varsa (missing) kullanıcı uyarısına rağmen devam ediyoruz (df'i olduğu gibi veriyoruz)
                    X = df[required_cols] if not missing else df
                    
                    # ID kolonunu modelden uzak tut
                    if 'id' in df.columns:
                        X = df.drop(columns=['id'], errors='ignore')
                    if 'Id' in df.columns:
                        X = df.drop(columns=['Id'], errors='ignore')
                        
                    clusters = model.predict(X)
                    df['Predicted_Cluster'] = clusters
                    
                    st.success("İşlem Tamamlandı!")
                    
                    # Visualization (İlk 3 feature varsa 3D çiz)
                    if len(df.columns) >= 3:
                        try:
                            fig = px.scatter_3d(
                                df.head(1000), 
                                x=df.columns[1], y=df.columns[2], z=df.columns[3],
                                color='Predicted_Cluster',
                                title="3D Scatter of First 3 Features (Sample)"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.write("3D görselleştirme için uygun veri formatı bulunamadı.")
                    
                    # Submission format
                    sub = df.copy()
                    
                    # Id kolonu standardizasyonu
                    if 'id' in sub.columns:
                        sub = sub.rename(columns={'id': 'Id'})
                    elif 'Id' not in sub.columns:
                        sub['Id'] = sub.index
                    
                    # Sadece Id ve Predicted_Cluster'ı al
                    try:
                        submission = sub[['Id', 'Predicted_Cluster']].rename(columns={'Predicted_Cluster': 'Predicted'})
                        csv = submission.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Submission CSV", csv, "submission.csv", "text/csv")
                    except KeyError as e:
                        st.error(f"CSV oluşturulurken kolon hatası: {e}. Tüm tabloyu indirilebilir yapıyorum.")
                        csv_full = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Full Results", csv_full, "results.csv", "text/csv")
                    
            except Exception as e:
                st.error(f"Dosya işleme hatası: {e}")

if __name__ == "__main__":
    main()