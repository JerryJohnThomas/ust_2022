import streamlit as st
from img2 import image_streamlit
from video1 import video_streamlit

def main_page():
    st.markdown("# Main page 🎈")
    # st.sidebar.markdown("# Main page 🎈")

def Image():
    image_streamlit()

def page3():
    video_streamlit()

page_names_to_funcs = {
    "Image": Image,
    "Video": page3,
}

# st.title("ALPR")
st.markdown("# ALPR")
# selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
selected_page = st.selectbox("Select Media Mode", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


# def footer():
footer="""<style>
a:link , a:visited{
color: gray;
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: black;
background-color: transparent;
text-decoration: none;
}

.footer {

position: fixed;
right: 0;
bottom: 0;
width: 100%;
background-color: transparent;
color: black;
text-align: center;
display:flex;
flex-direction:row;
justify-content:flex-end;
padding:0 15px 0 0 ;
font-size:70px;
}
</style>̥
<div class="footer">
<p style='align-content:flex-end;'>Developed with <span style="color:red">❤</span> by 
    <a style='display: inline;' href="https://www.linkedin.com/in/jasir721/?originalSubdomain=in" target="_blank">Jasir</a>
    &
    <a style='display: inline;' href="https://www.linkedin.com/in/jerry-john-thomas-4787601b5/" target="_blank">Jerry</a>
</p>
</div>
"""


st.markdown(footer,unsafe_allow_html=True)


st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
