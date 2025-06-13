import os
import streamlit as st
import pandas as pd
import sweetviz as sv
import chardet
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai

# ========== CONFIGURATION ==========
# Set Gemini API key
genai.configure(api_key="AIzaSyBcdUlv3ULcSlbGU0HoK0HGCaRXj72WoPs") 

# Streamlit page config
st.set_page_config(page_title="Auto Analyzer v5 Enterprise", layout="wide")

# ========== FILE LOADER MODULE ==========

def load_file(uploaded_file):
    try:
        filename = uploaded_file.name.lower()

        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif filename.endswith(('.csv', '.tsv', '.txt')):
            rawdata = uploaded_file.read()
            result = chardet.detect(rawdata)
            encoding = result['encoding']
            uploaded_file.seek(0)
            if filename.endswith('.tsv'):
                df = pd.read_csv(uploaded_file, encoding=encoding, sep='\t')
            else:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                except:
                    df = pd.read_csv(uploaded_file, encoding=encoding, sep=';')
        else:
            raise ValueError("Unsupported file format.")

        if df.empty:
            raise ValueError("File loaded but contains no data.")

        return df, None
    except Exception as e:
        return None, f"Lỗi đọc file: {str(e)}"

# ========== MISSING VALUE HANDLER ==========

def handle_missing(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# ========== AI REPORT MODULE ==========

def analyze_data_ai(df):
    prompt = f"""
    Tôi có bảng dữ liệu dưới dạng CSV sample như sau:
    {df}

    Bạn là chuyên gia phân tích dữ liệu. Hãy viết một báo cáo phân tích chuyên sâu từ dữ liệu đã cung cấp.
    - Báo cáo dự trên các nội dung đang có
    - Phát hiện các vấn đề tiềm ẩn (nếu có).
    - Đề xuất hướng phân tích tiếp theo hoặc các biến quan trọng.
    - Viết bằng tiếng Việt phân tích chuyên nghiệp như data science.
    - Nếu nội dung là về thương mại thì hãy đưa ra chiến lược marketing
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

# ========== MAIN APP UI ==========

st.title("🚀 Tự động phân tích dữ liệu và viết báo cáo")

# Sidebar Upload
with st.sidebar:
    st.header("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Chọn file (CSV, Excel, TXT, TSV)", type=['csv', 'xlsx', 'xls', 'txt', 'tsv'])

# Load file
if uploaded_file is not None:
    df, error = load_file(uploaded_file)
    if error:
        st.error(error)
        st.stop()

    # Handle missing
    df = handle_missing(df)

    # Tab layout
    tab1, tab2, tab3 = st.tabs(["📊 Preview & Stats", "📈 Visualization", "🧠 AI Report"])

    # ========= TAB 1: PREVIEW =========
    with tab1:
        st.subheader("📊 Dữ liệu sau khi xử lý")
        st.write(df.head())
        st.subheader("📈 Thống kê mô tả")
        st.write(df.describe(include='all'))
        # Optional: Sweetviz EDA
        if st.button("🔬 Sinh EDA Report (Sweetviz)", key="eda"):
            with st.spinner("Đang tạo Sweetviz..."):
                os.makedirs("reports", exist_ok=True)
                eda_report_path = "reports/eda_report.html"
                report = sv.analyze(df)
                report.show_html(eda_report_path)

                with open(eda_report_path, 'r', encoding='utf-8') as f:
                    html_data = f.read()
                st.components.v1.html(html_data, height=800, scrolling=True)
    # ========= TAB 2: VISUALIZATION =========
    with tab2:
        st.subheader("📊 Biểu đồ phân tích tự động")
        numeric_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()

        if len(numeric_cols) >= 2:
            st.write("🔵 Heatmap - Correlation")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        if len(numeric_cols) >= 1:
            st.write("🔵 Histogram các cột số")
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f"Histogram - {col}")
                st.pyplot(fig)

    # ========= TAB 3: AI REPORT =========
    with tab3:
        st.subheader("🧠 Sinh báo cáo AI (Gemini)")
        if st.button("🧠 Tạo báo cáo"):
            with st.spinner("Đang phân tích bằng Gemini AI..."):
                try:
                    report = analyze_data_ai(df)
                    st.write(report)
                except Exception as e:
                    st.error(f"Lỗi khi gọi AI: {e}")

else:
    st.info("📂 Vui lòng upload dữ liệu từ Sidebar.")
