import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import time
import tempfile
from video_processing import video_to_set_process

def video_streamlit():
    # st.markdown("### Video ALPR")
    uploaded_file = st.file_uploader("Choose a video file", type=["mkv","mp4"])

    col1, col2 = st.columns(2)

    if uploaded_file is not None:

        with st.spinner('Processing ...'):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            inp_video = cv2.VideoCapture(tfile.name)
            # your code here.

            # instructions to Jasir, so inp_video is hopefully the video so just call it here.
            out_set=video_to_set_process(inp_video)

            time.sleep(2)
            st.success('Done ust!')

            st.download_button('Download CSV', out_set, 'text/csv')
            st.download_button('Download as text', out_set)


   

