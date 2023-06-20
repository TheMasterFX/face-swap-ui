
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

import gradio as gr

theme = gr.themes.Default(
    font=['Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
).set(
    border_color_primary='#c5c5d2',
    button_large_padding='6px 12px',
    body_text_color_subdued='#484848',
    background_fill_secondary='#eaeaea'
)

face_coords = [0,0]
faces = []
app = FaceAnalysis(name='buffalo_l')

def select_handler(img, evt: gr.SelectData):
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    cropped_image = []
    face_index = 0
    print("Coords: ", evt.index[1],evt.index[0])
    for face in faces:
        box = face.bbox.astype(np.int32)
        face_index = face_index + 1 
        if point_in_box((box[0], box[1]),(box[2],box[3]),(evt.index[0],evt.index[1])) == True:
            print("True ", face_index)
            cropped_image = img[box[1]:box[3],box[0]:box[2]]
    return cropped_image

def greet(coords):
    return coords     

def img_load():
    print("Load")

def point_in_box(bl, tr, p) :
   if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
      return True
   else :
      return False
   
def get_faces(img):
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    print("Faces: ", len(faces))
    
def create_interface():
    title = 'Face Swap UI'
    with gr.Blocks(analytics_enabled=False, title=title) as face_swap_ui:
        with gr.Tab("Swap Face Image"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("Input Image:")
                    image_input = gr.Image(label='Input Image (Click to select a face)')
                    
                    get_faces_button = gr.Button("Analyze")
                    with gr.Row():
                        selected_face = gr.Image(label='Face to swap', interactive=False)
                        swap_face = gr.Image(label='Swap Face')
                        image_input.select(select_handler, image_input, selected_face)
                    gr.Slider(value=50, label="Tolerance", info="Tolerance", interactive=True)
                    get_faces_button.click(fn=get_faces, inputs=image_input)
                with gr.Column():
                    image_output = gr.Image(label='Output Image')
            swap_button = gr.Button("Swap")
            #swap_button.click(fn=greet, inputs=text_output, outputs=text_output)
        with gr.Tab("Swap Face Video"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")


    face_swap_ui.launch()



if __name__ == "__main__":

#    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
#    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

    create_interface()  