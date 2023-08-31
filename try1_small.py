import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
import os
import csv



#program->
def scale(eigen_img, s):
        scale_percent = s
        width = int(eigen_img.shape[1] * scale_percent / 100)
        height = int(eigen_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        eigen_img = cv2.resize(eigen_img, dim)

        return eigen_img

def ready(uploaded_file, image_size, PATH): #読み込み+画像の縮尺
    if uploaded_file is None:
        st.write("ファイル名をアップロードしてください。")
        return
    
    imagename=uploaded_file.name
    #PATH = os.getcwd()
    #st.write(PATH + "/" + imagename)

    #st.write(PATH)
    #st.write(os.listdir(PATH + "/" +"download" ))

    if os.path.isfile(PATH+"/"+ imagename):
        st.write("file OK")    

    #eigen_img = cv2.imread(PATH+"/"+ imagename)
    #eigen_img = cv2.imread(imagename)
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
        eigen_img = cv2.imdecode(file_bytes, 1)


    if eigen_img is None:
        exit(-1)
    harris_img = eigen_img.copy()
    fast_img = eigen_img.copy()

    b = image_size
    s = int(b)

    eigen_img=scale(eigen_img, s)

    #img_flip_ud_lr=cv2.flip(eigen_img, -1)
    img_flip_ud_lr = eigen_img
    # cv2.imwrite(img_flip_ud_lr)

    return img_flip_ud_lr


def generate_binary(img_flip_ud_lr): #二値化
    if img_flip_ud_lr is None:
        print("Error:img_flip_ud_lr is None")
        return
    
    gray_img = cv2.cvtColor(img_flip_ud_lr, cv2.COLOR_BGR2GRAY)
    ret, img_thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    st.image(img_thresh) 
    return gray_img


# setup working directory
PATH = os.getcwd()

#サイト
st.title("Vorography") #title

st.markdown(""" <style> .font {
    font-0size:35px ; font-family: 'Cooper Black'; color: #ffffff;} 
    </style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Upload your photo here...</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("file Upload", type=(['jpg', 'png', 'jpeg'])) #画像のアップロード
image_size = st.text_input("Set the size of the image...(画像の大きさを入力してください)") #画像のサイズ
st.caption("Set the scale as a percentage  ex:80%->80(縮尺をパーセンテージで入れてください 例:80%->80)")#入力例を入れる
#image_output =st.text_input("Set the file name...(ファイル名を決めてください)") #画像出力パス
#st.caption("file name.jpg or png or jpeg(ファイル名.jpg or png or jpeg )")
#csv_output =st.text_input("Set the csv file name...(csvファイル名を決めてください)") #csvパス
#st.caption("file name.csv(ファイル名.csv)")

switch1=st.checkbox("Next")

if switch1:
    img_flip_ud_lr=ready(uploaded_file, image_size, PATH)
    if img_flip_ud_lr is not None:
        st.image(img_flip_ud_lr)


    tab1, tab2= st.tabs(["Point(特徴点)", "Binarization(二値化)"])
    
    with tab2:
        st.header("Binarization(二値化)")
        gray_image = generate_binary(img_flip_ud_lr)
    

    with tab1:
        st.header("Features point extraction(特徴点抽出)")
        feature=st.slider("Determine the number of points...(点の数を決めてください)", 0 ,300 ,50) # スライダー    
        st.caption("Once you have determined the number of dots, be sure to check the box(点の数を決めたら、必ずチェックを入れて下さい)")
        switch2=st.checkbox("Done")
        if switch2:
            #generate_point(img_flip_ud_lr, gray_image)

            st.caption("(ボタンを押したら、ボロノイ図が生成されます)")
            if st.button("Next"):
                #generate_voronoi_diagram(num_feature_points)
                #st.pyplot() #ボロノイ図の表示
        
    

                tab1, tab2 = st.tabs(["Coloring(着色)","(重ね合わせ)"])

                with tab2:
                    st.header("(重ね合わせ)")
                #    #st.image()

                with tab1:
                    st.header("Coloring(着色)")
    
               #    st.subheader("Color sample(色見本)")
                #    #色見本の表示
                    data={'color':['Red(赤)', 'Reddish-purple(赤紫)', 'Pink(ピンク)', 'Purple(紫)', 'Blue(青)', 'Bluish-purple(青紫)', 'Aqua(水色)', 'Bluish-green(青緑)', 'Green(緑)', 'Yellow-green(黄緑)', 'Yellow(黄色)', 'Orange(オレンジ)', 'Brown(茶色)', 'Black(黒色)', 'Grey(灰色)'],   
                     'Red':['230', '235', '245', '136', '0', '103', '188', '0', '62', '184', '255', '238', '150', '43', '125'],
                     'Green' :['0', '110', '178', '72', '149', '69', '226', '164', '179', '210', '217', '120', '80', '43', '125'],
                     'Blue' :['51', '165', '178', '152', '217', '60','232', '151', '112', '0', '0', '0', '66', '43', '125']
                     }
    
    
                    df = pd.DataFrame(data)
                    st.table(df) # 静的な表

                    st.caption("Colors other than those in the color samples can be used.(色見本以外の色も使用可能です)")

                    r=st.slider("choose red",0,256,128,1)
                    g=st.slider("choose green",0,256,128,1)
                    b=st.slider("choose blue",0,256,128,1)

                    st.caption("Once you decide on a color, be sure to check the box!(色を決めたら、チェックして下さい)")
                #    check3=st.checkbox("Done")
                #    if check3:
                #        generate_color(points, xlim, ylim, region_colors)
                #        st.pyplot()