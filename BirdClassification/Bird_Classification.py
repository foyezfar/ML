import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.utils import image_utils

model = load_model('BC.h5',compile=False)
lab = {0: 'COCKATOO', 1: 'CROW', 2: 'EMU', 3: 'OSTRICH', 4: 'PEACOCK', 5: 'WOOD DUCK'}

def processed_img(img_path):
    img=image_utils.load_img(img_path,target_size=(224,224,3))
    img=image_utils.img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res

def run():
    st.title("Birds Species Classification")
    img_file = st.file_uploader("Choose an Image of Bird", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = processed_img(save_image_path)
            st.success("Predicted Bird is: "+result)
run()                