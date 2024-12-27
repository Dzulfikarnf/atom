import streamlit as st
import joblib
import pandas as pd

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    return joblib.load('random_forest_model.joblib')

model = load_model()

# Fungsi untuk membuat prediksi
def predict_stability(_1nz, N, Z, A, MASSEXCESS, AtomicMass, BE, BE_uncer):
    # Perhitungan otomatis untuk N/Z dan BE/A
    NZ_ratio = 0 if Z == 0 else N / Z
    BE_per_A = 0 if A == 0 else BE / A

    # Kolom fitur
    feature_columns = ['1nz', 'N', 'Z', 'A', 'EL', 'MASSEXCESS', 
                       'AtomicMass', 'BE', 'BE_uncer', 'NZ_ratio', 'BE_per_A']

    # Dataframe input
    input_data = pd.DataFrame([[ _1nz, N, Z, A, 0, MASSEXCESS, 
                                 AtomicMass, BE, BE_uncer, NZ_ratio, BE_per_A]], 
                              columns=feature_columns)

    # Prediksi
    prediction = model.predict(input_data)

    # Mapping hasil prediksi ke kategori stabilitas
    stability_mapping = {
        0: "Sangat Stabil",
        1: "Stabil",
        2: "Radioaktif",
        3: "Sangat Radioaktif"
    }

    return stability_mapping[prediction[0]], prediction[0]

# UI Streamlit
st.title("Prediksi Stabilitas Inti Atom")
st.subheader("Model Machine Learning untuk Memprediksi Stabilitas Inti Atom")
st.markdown("""
Aplikasi ini memanfaatkan model machine learning untuk memprediksi stabilitas inti atom berdasarkan fitur yang dimasukkan.
Silakan masukkan nilai-nilai berikut untuk mendapatkan hasil prediksi:
""")

# Form input
with st.form("prediction_form"):
    st.write("### Masukkan Fitur:")
    _1nz = st.number_input("Masukkan nilai 1nz:", value=0.0, step=0.01)
    N = st.number_input("Masukkan jumlah neutron (N):", value=0.0, step=0.01)
    Z = st.number_input("Masukkan jumlah proton (Z):", value=0.0, step=0.01)
    A = st.number_input("Masukkan nomor massa (A):", value=0.0, step=0.01)
    MASSEXCESS = st.number_input("Masukkan nilai MASSEXCESS:", value=0.0, step=0.01)
    AtomicMass = st.number_input("Masukkan nilai AtomicMass:", value=0.0, step=0.01)
    BE = st.number_input("Masukkan nilai BE (Energi Ikat):", value=0.0, step=0.01)
    BE_uncer = st.number_input("Masukkan nilai BE_uncer:", value=0.0, step=0.01)
    
    # Tombol prediksi
    submit = st.form_submit_button("Prediksi")

# Proses prediksi
if submit:
    try:
        stability, class_label = predict_stability(_1nz, N, Z, A, MASSEXCESS, AtomicMass, BE, BE_uncer)
        st.write("### Hasil Prediksi")
        st.success(f"**Stabilitas:** {stability}")
        st.info(f"**Kelas Prediksi:** {class_label}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
