import cv2
import streamlit as st
from PIL import Image,ImageEnhance
import numpy as np
import time
import imutils
import numpy as np
from base64 import b64decode
import mediapipe as mp


#Haar Cascade Face Detection
Haar_Cascade_start = time.time()
face_cascade = cv2.CascadeClassifier('frecog/haarcascade_frontalface_default.xml')
def haar_cascade_detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
    return img,faces
Haar_Cascade_end = time.time()

#SSD Face Detection
SSD_start = time.time()
def ssd_detect_faces(our_image):
    #Load the pre-trained face detection network model
    prototxt = 'deploy.prototxt'
    model = 'res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    new_img = np.array(our_image.convert('RGB'))
    image = cv2.cvtColor(new_img,1)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        # Extract the confidence ( probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # Filter out weak detections
        if confidence > 0.5:
        # Compute the (x, y) coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Draw rectangle around detected faces
            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
    return image
SSD_end = time.time()

#Mediapipe Face Detection
Mediapipe_start = time.time()
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detect_faces(our_image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        new_img = np.array(our_image.convert('RGB'))
        image = cv2.cvtColor(new_img,1)
        #Process image with MediaPipe Face Detection
        results = face_detection.process(image)
        annotated_image = image.copy()
        # Draw rectangle around detected faces
        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection, 
                                      bbox_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3))
            faces = results.detections
    return annotated_image, faces
Mediapipe_end = time.time()

#Effects
def cartonize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Edges
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    #Color
    color = cv2.bilateralFilter(img, 9, 300, 300)
    #Cartoon
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def cannize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny


#Streamlit App
def main():
    """Face Detection App"""
    st.title("Face Detection App")
    st.text("Projektni zadatak - Karla Fehir")
    activities = ["Haar Cascade", "Mediapipe", "Single Shot Detector"]
    choice = st.sidebar.selectbox("Select Activty",activities)

    image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

    if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image)

    if choice == 'Haar Cascade':
        st.subheader("Haar Cascade Face Detection")
        if st.button("Process"):
                result_img,result_faces = haar_cascade_detect_faces(our_image)
                st.image(result_img)
                st.success("Found {} faces".format(len(result_faces)))
                st.write("Haar Cascade time elapsed: ", (Haar_Cascade_end-Haar_Cascade_start)/1000)

    if choice == 'Mediapipe':
        st.subheader("Mediapipe Face Detection")
        if st.button("Process"):
                result_img,result_faces = mediapipe_detect_faces(our_image)
                st.image(result_img)
                st.success("Found {} faces".format(len(result_faces)))
                st.write("Mediapipe time elapsed: ", (Mediapipe_end-Mediapipe_start)/1000)

    if choice == 'Single Shot Detector':
        st.subheader("Single Shot Detector Face Detection")
        if st.button("Process"):
            result_img = ssd_detect_faces(our_image)
            st.image(result_img)
            #st.success("Found {} faces".format(len(result_faces)))
            st.write("SSD time elapsed: ", (SSD_end-SSD_start)/1000)
   

    enhance_type = st.sidebar.radio("Enhance Type",["Original","Gray-Scale","Contrast","Brightness",
                                                    "Blurring","Cartonize","Cannize"])
    if enhance_type == 'Gray-Scale':
        new_img = np.array(our_image.convert('RGB'))
        img = cv2.cvtColor(new_img,1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # st.write(new_img)
        st.image(gray)

    elif enhance_type == 'Cartonize':
        result_img = cartonize_image(our_image)
        st.image(result_img)

    elif enhance_type == 'Cannize':
        result_canny = cannize_image(our_image)
        st.image(result_canny)

    elif enhance_type == 'Contrast':
        c_rate = st.sidebar.slider("Contrast",0.5,3.5)
        enhancer = ImageEnhance.Contrast(our_image)
        img_output = enhancer.enhance(c_rate)
        st.image(img_output)

    elif enhance_type == 'Brightness':
        c_rate = st.sidebar.slider("Brightness",0.5,3.5)
        enhancer = ImageEnhance.Brightness(our_image)
        img_output = enhancer.enhance(c_rate)
        st.image(img_output)

    elif enhance_type == 'Blurring':
        new_img = np.array(our_image.convert('RGB'))
        blur_rate = st.sidebar.slider("Brightness",0.5,3.5)
        img = cv2.cvtColor(new_img,1)
        blur_img = cv2.GaussianBlur(img,(11,11),blur_rate)
        st.image(blur_img)

if __name__ == '__main__':
        main()
