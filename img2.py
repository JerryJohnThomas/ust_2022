import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import base64
from image_processing import label_input_img

def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
    # im
    # img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB)
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a style="text-align: center; text-decoration:none;background-color:#0068c9;color:white;border:1px solid ;padding:5px 8px;borderRadius:6px;"  href="data:file/jpg;base64,{img_str}" download ="result.jpg">Download result</a>'
	return href


def image_streamlit():
    # st.markdown("### Image ALPR")
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")
    col1, col2 = st.columns(2)


    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        opencv_image = cv2.cvtColor( opencv_image, cv2.COLOR_BGR2RGB)
        



        # Now do something with the image! For example, let's display it:
        col1.image(opencv_image, caption=f"Input Image")

        # reconvverting
        opencv_image, out_label = cv2.cvtColor( opencv_image, cv2.COLOR_RGB2BGR)


        # instructions to Jasir, make output_img as the labelled image.
        output_img=label_input_img(opencv_image) # add function here



        col2.image(output_img, caption=f"Output")
        result = Image.fromarray(output_img)
        st.markdown(out_label)
        st.markdown(get_image_download_link(result), unsafe_allow_html=True)


