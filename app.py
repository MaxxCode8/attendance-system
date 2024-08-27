import streamlit as st
from deepface import DeepFace
from retinaface import RetinaFace
import pandas as pd
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime

# Global variables
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

def main():
    st.set_page_config(page_title="Classroom Attendance System", layout="wide")

    st.title("Classroom Attendance System")

    tab1, tab2, tab3 = st.tabs(["Take Attendance", "Manage Students", "Analytics"])

    with tab1:
        take_attendance()

    with tab2:
        manage_students()

    with tab3:
        show_analytics()

def take_attendance():
    st.header("Take Attendance")
    
    # Camera input
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer is not None:
        # Read image from buffer
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Detect faces using RetinaFace
        faces = RetinaFace.detect_faces(cv2_img)
        
        if isinstance(faces, dict):
            st.write(f"Detected {len(faces)} faces.")
            
            # Process each detected face
            for face_idx, face_info in faces.items():
                facial_area = face_info['facial_area']
                x, y, w, h = facial_area
                
                # Extract face from image
                face_img = cv2_img[y:h, x:w]
                
                # Save face temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    cv2.imwrite(tmp_file.name, face_img)
                    
                    # Recognize face using DeepFace
                    try:
                        result = DeepFace.find(tmp_file.name, db_path=KNOWN_FACES_DIR, enforce_detection=False)
                        if not result[0].empty:
                            student_name = os.path.splitext(os.path.basename(result[0].iloc[0]['identity']))[0]
                            st.write(f"Recognized: {student_name}")
                            mark_attendance(student_name)
                        else:
                            st.write("Unknown face detected.")
                    except Exception as e:
                        st.write(f"Error in face recognition: {str(e)}")
                    
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
        else:
            st.write("No faces detected.")

def mark_attendance(student_name):
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")
    
    if not os.path.isfile(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
    else:
        df = pd.read_csv(ATTENDANCE_FILE)
    
    if not ((df['Name'] == student_name) & (df['Date'] == date)).any():
        new_row = pd.DataFrame({'Name': [student_name], 'Date': [date], 'Time': [time]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        st.success(f"Attendance marked for {student_name}")
    else:
        st.info(f"Attendance already marked for {student_name} today")

def manage_students():
    st.header("Manage Students")
    
    # File uploader for student photo
    student_name = st.text_input("Student Name")
    uploaded_file = st.file_uploader("Choose a photo", type=['jpg', 'jpeg', 'png'])
    
    if st.button("Register Student") and student_name and uploaded_file:
        register_student(student_name, uploaded_file)

def register_student(name, photo):
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
    
    file_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    with open(file_path, "wb") as f:
        f.write(photo.getbuffer())
    
    st.success(f"Student {name} registered successfully!")

def show_analytics():
    st.header("Analytics Dashboard")
    
    if os.path.isfile(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        
        # Display overall attendance
        st.subheader("Overall Attendance")
        st.dataframe(df)
        
        # Student-wise attendance
        st.subheader("Student-wise Attendance")
        student_attendance = df['Name'].value_counts()
        st.bar_chart(student_attendance)
        
        # Date-wise attendance
        st.subheader("Date-wise Attendance")
        date_attendance = df['Date'].value_counts()
        st.line_chart(date_attendance)
    else:
        st.info("No attendance data available yet.")

if __name__ == "__main__":
    main()
