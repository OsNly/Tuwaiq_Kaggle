# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# App title
st.set_page_config(page_title="Prediction App", layout="wide")

st.title("Program Completion Prediction for Tuwaiq Academy")
st.markdown("---")
st.markdown(" Note: test data should be in the same format as the training data.")

# Load model and encoders
model = joblib.load('final_model_pipeline1.pkl')
encoders = joblib.load('label_encoders.pkl')

# === File uploader ===
st.sidebar.header("Upload Your Test File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# === Automatically use default CSV if nothing uploaded ===
if uploaded_file is None:
    st.warning("You can make predicitons on 'test.csv' or upload a new file on the left side.")
    uploaded_file = 'test.csv'

# === Utility Functions ===
def clean_light(text):
    text = str(text)
    text = text.strip()
    text = re.sub("[^ء-يA-Za-z0-9 ]+", "", text)
    text = re.sub("\s+", " ", text)
    text = re.sub(r'\bال', '', text)
    return text.lower()

def cleantext(text):
    text = str(text).split()
    text1 = ""
    for txt in text:
        txt = clean_light(txt)
        text1 = text1 + txt + ' '
    return text1.strip()

def infer_college(speciality):
    if pd.isna(speciality):
        return None
    s = speciality.lower()
    if any(word in s for word in ['نظم معلومات','علم بيانات','information system','data science','network','information systems','اصطناعي','مواقع','برمجيات','سيبراني','computer','دعم فني','سيبراني', 'نظم معلومات','it','شبكات', 'برمجه','تقنيه','technology','software','امن معلومات' ,'cis', 'حاسب','حاسوبيه']):
        return 'تكنولوجيا الاتصالات والمعلومات'
    elif any(word in s for word in ['اداره', 'اقتصاد','اعمال', 'business', 'mis','management information system']):
        return 'إدارة الأعمال'
    elif any(word in s for word in ['هندسه','enginering']):
        return 'الهندسة'
    elif any(word in s for word in ['ترجمه', 'انجليزيه','لغه', 'لسانيات','تاريخ']):
        return 'الآداب والترجمة'
    elif any(word in s for word in ['قانون', 'شريعه', 'حقوق','وقضايا']):
        return 'القانون أو الشريعة'
    elif any(word in s for word in ['تربيه', 'تعليم']):
        return 'التربية'
    elif any(word in s for word in ['فيزياء','physics','جيولوجيا', 'كيمياء', 'احياء', 'رياضيات', 'علوم', 'bio', 'chemistry', 'biology', 'math']):
        return 'كلية العلوم'
    elif any(word in s for word in ['طب','صيدلي','تمريض', 'امراض','جراحه', 'طبيه',  'طب بشري', 'طب عام',"اشعه" ,'medicin', 'surgery']):
        return 'الطب'
    else:
        return 'كلية أخرى'

# === Main File Processing ===
if uploaded_file:
    df_original = pd.read_csv(uploaded_file) if isinstance(uploaded_file, str) else pd.read_csv(uploaded_file)
    df = df_original.copy()
    registration = pd.read_csv('registration.csv')

    st.subheader("Uploaded Data Preview")
    st.dataframe(df_original.head())

    df['Unified_Score_Percentage'] = (
        (df['University Degree Score'] / df['University Degree Score System']) * 100).round(2)

    df.loc[df['College'].isna(), 'Education Speaciality'] = df.loc[df['College'].isna(), 'Education Speaciality'].apply(lambda x: cleantext(x))
    df['College_Filled'] = df['College']
    df.loc[df['College'].isna(), 'College_Filled'] = df.loc[df['College'].isna(), 'Education Speaciality'].apply(infer_college)

    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Program Sub Category Code'] = df['Program Sub Category Code'].fillna(df['Program Main Category Code'])
    df['Program Skill Level'] = df['Program Skill Level'].fillna('غير معروف')
    df['Level of Education'] = df['Level of Education'].fillna('غير معروف')
    df['Employment Status'] = df['Employment Status'].fillna('غير معروف')
    Percentage_mean = df['University Degree Score System'].mean()
    df['Unified_Score_Percentage'] = df['Unified_Score_Percentage'].fillna(Percentage_mean)
    df['Home City'] = df['Home City'].fillna(df['Home City'].mode()[0])

    df = df.merge(registration[['Student ID', 'Total Regestration']], on='Student ID', how='left')

    df = df.drop(columns=[
        'Program Start Date', 'Program End Date',
        'Technology Type', 'Education Speaciality', 'University Degree Score System',
        'Job Type', 'Still Working', 'College', 'University Degree Score'
    ], errors='ignore')

    for col in df.select_dtypes(include='object').columns:
        if col in encoders:
            known_labels = set(encoders[col].classes_)
            df[col] = df[col].apply(lambda x: x if x in known_labels else 'unknown')
            if 'unknown' not in encoders[col].classes_:
                encoders[col].classes_ = np.append(encoders[col].classes_, 'unknown')
            df[col] = encoders[col].transform(df[col])

    if st.button("Predict"):
        try:
            probs = model.predict_proba(df)[:, 1]
            predictions = (probs > 0.47).astype(int)

            df_result = df_original.copy()
            df_result['Prediction'] = predictions
            df_result['Probability_Not_Completed'] = probs

            st.subheader("Prediction Results")
            st.dataframe(df_result)

            csv = df_result.to_csv(index=False)
            st.download_button("Download Results", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# === Manual Input Prediction Section ===
st.markdown("## Or Manually Enter Values for Prediction")

with st.form("manual_input_form"):
    st.markdown("### Enter Input Values")

    age = st.number_input("Age", min_value=10, max_value=70, value=25)
    gender = st.selectbox("Gender", options=["ذكر", "أنثى"])
    home_region = st.selectbox(
    "Home Region", 
    options=[
        "منطقة الرياض",
        "منطقة مكة المكرمة",
        "المنطقة الشرقية",
        "منطقة المدينة المنورة",
        "منطقة عسير",
        "منطقة القصيم",
        "منطقة جازان",
        "منطقة تبوك",
        "منطقة الباحة",
        "منطقة حائل",
        "منطقة نجران",
        "منطقة الحدود الشمالية",
        "منطقة الجوف"
        ]
    )
    home_city = st.selectbox(
        "Home City",
        options=[
            "الرياض",
            "مكة المكرمة",
            "الشرقية",
            "المدينة المنورة",
            "عسير",
            "القصيم",
            "جازان",
            "تبوك",
            "الباحة",
            "حائل",
            "نجران",
            "الحدود الشمالية",
            "الجوف"
        ]
    )
    main_category = st.selectbox(
        "Program Main Category Code",
        options=[
            "CAUF",
            "PCRF",
            "APMR",
            "TOSL",
            "GRST",
            "ABIR",
            "INFA",
            "SERU",
            "DTFH",
            "QWLM"
        ]
    )
    sub_category = st.selectbox(
        "Program Sub Category Code",
        options=[
            "SWPS",
            "PCRF",
            "SRTA",
            "INFA",
            "TOSL",
            "APMR",
            "CAUF",
            "CRDP",
            "ERST",
            "KLTM",
            "ABIR",
            "QTDY",
            "ASCW",
            "DTFH",
            "QWLM"
        ]
    )
    skill_level = st.selectbox("Program Skill Level", options=["غير معروف", "متوسط", "مبتدئ", "متقدم"])
    presentation_method = st.selectbox("Program Presentation Method", options=["حضوري", "عن بعد"])
    program_days = st.number_input("Program Days", min_value=1, max_value=300, value=10)
    completed_degree = st.selectbox("Completed Degree", options=["نعم", "لا"])
    education_level = st.selectbox("Level of Education", options=["ثانوي", "الدبلوم", "البكالوريوس", "الماجستير", "الدكتوراه"])
    employment_status = st.selectbox("Employment Status", options=["طالب", "خريج", "موظف", "غير موظف", "غير معروف"])
    unified_score = st.number_input("Unified Score Percentage", min_value=0.0, max_value=100.0, value=75.0)
    college_filled = st.selectbox("College (Filled)", options=[
        "تكنولوجيا الاتصالات والمعلومات",
        "كلية أخرى",
        "كلية العلوم",
        "الأعمال والإدارة والقانون",
        "العلوم الطبيعية والرياضيات والإحصاء",
        "إدارة الأعمال",
        "الهندسة والتصنيع والبناء",
        "الفنون والعلوم الإنسانية",
        "العلوم الاجتماعية والصحافة والإعلام",
        "الطب",
        "التعليم",
        "التربية",
        "الآداب والترجمة",
        "الصحة والرفاة",
        "الهندسة",
        "القانون أو الشريعة",
        "البرامج والمؤهلات العامة"
    ])
    total_registration = st.number_input("Total Registration", min_value=1, max_value=110, value=5)

    submit_button = st.form_submit_button("Predict Manually")

    if submit_button:
        try:
            input_dict = {
                "Student ID": ["placeholder"],
                "Age": [age],
                "Gender": [gender],
                "Home Region" : [home_region],
                "Home City": [home_city],
                "Program ID": ["placeholder"],
                "Program Main Category Code": [main_category],
                "Program Sub Category Code": [sub_category],
                "Program Skill Level": [skill_level],
                "Program Presentation Method": [presentation_method],
                "Program Days": [program_days],
                "Completed Degree": [completed_degree],
                "Level of Education": [education_level],
                "Employment Status": [employment_status],
                "Unified_Score_Percentage": [unified_score],
                "College_Filled": [college_filled],
                "Total Regestration": [total_registration],
                
                
                
                
            }

            input_df = pd.DataFrame(input_dict)
            for col in input_df.select_dtypes(include='object').columns:
                if col in encoders:
                    known_labels = set(encoders[col].classes_)
                    input_df[col] = input_df[col].apply(lambda x: x if x in known_labels else 'unknown')
                    if 'unknown' not in encoders[col].classes_:
                        encoders[col].classes_ = np.append(encoders[col].classes_, 'unknown')
                    input_df[col] = encoders[col].transform(input_df[col])

            prob = model.predict_proba(input_df)[0][1]
            prediction = int(prob > 0.47)

            st.success(f"**Prediction:** {'he/she will Not Complete the program' if prediction == 1 else 'he/she will Complete the program'}")
            st.info(f"Probability of Not Completing: {prob:.2f}")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
