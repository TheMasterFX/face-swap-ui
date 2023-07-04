import numpy as np
import cv2
import os
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

def add_bbox_padding(bbox, margin=5):
    return [
        bbox[0] - margin,
        bbox[1] - margin,
        bbox[2] + margin,
        bbox[3] + margin]


def select_handler(img, evt: gr.SelectData): 
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    cropped_image = []
    face_index = -1
    sel_face_index = 0
    print("Coords: ", evt.index[0],evt.index[1])
    for face in faces:
        box = face.bbox.astype(np.int32)
        face_index = face_index + 1 
        if point_in_box((box[0], box[1]),(box[2],box[3]),(evt.index[0],evt.index[1])) == True:
            # print("True ", face_index)
            # print("Bbox org: ", box)
            # Add ~25% margin to the box so the face is recognized correctly
            margin = int((box[2]-box[0]) * 0.35)
            box = add_bbox_padding(box,margin)
            box = np.clip(box,0,None)
            print("Bbox exp: ", box)
            sel_face_index = face_index            
            cropped_image = img[box[1]:box[3],box[0]:box[2]]
    return cropped_image, sel_face_index

def point_in_box(bl, tr, p) :
   if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
      return True
   else:
      return False
   
def get_faces(img):
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    #boxed_faces = app.draw_on(img, faces)
    #for i in range(len(faces)):
    #    face = faces[i]
    #    box = face.bbox.astype(np.int32)
    #    cv2.putText(boxed_faces,'Face#:%d'%(i), (box[0]-1, box[3]+14),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)

    return img, len(faces)

def swap_face_fct(img_source,face_index,img_swap_face):
    faces = app.get(img_source)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    src_face = app.get(img_swap_face)
    src_face = sorted(src_face, key = lambda x : x.bbox[0])
    #print("index:",faces)
    res = swapper.get(img_source, faces[face_index], src_face[0], paste_back=True)
    return res

def swap_video_fct(video_path, output_path, source_face, destination_face, tolerance, preview=-1, progress=gr.Progress()):

    # Get the Destination Face parameters (the face which should be swapped)
    dest_face = app.get(destination_face)
    dest_face = sorted(dest_face, key = lambda x : x.bbox[0])

    if(len(dest_face) == 0):
        print("No dest face found")
        return -1
 
    dest_face_feats = []
    dest_face_feats.append(dest_face[0].normed_embedding)
    dest_face_feats = np.array(dest_face_feats, dtype=np.float32)

    # Get the source face parameters (the face that replaces the original)
    src_face = app.get(source_face)
    src_face = sorted(src_face, key = lambda x : x.bbox[0])
    if(len(src_face) == 0):
        print("No source face found")
        return -1
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    # Use the same tmp dir from gradio if no output path is set
    if(len(output_path) > 0):
        out_path = output_path
    else:
        out_path = os.path.dirname(video_path) + "/out.mp4"

    if preview == -1:
        for_range = range(frame_count)
        video_out = cv2.VideoWriter(out_path,fourcc,fps,(width,height))
    else:
        for_range = range(preview-1,preview)

    for i in for_range:
        progress(i/frame_count, desc="Processing")
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all faces in the current frame
        faces = app.get(frame)
        faces = sorted(faces, key = lambda x : x.bbox[0])
        # No face in Scene => copy input frame

        if(len(faces) > 0):
            feats = []
            for face in faces:
                feats.append(face.normed_embedding)
            feats = np.array(feats, dtype=np.float32)
            sims = np.dot(dest_face_feats, feats.T)
            print(sims)
            # find the index of the most similar face
            max_index = np.argmax(sims)
            print("Sim:", max_index)
            if(sims[0][max_index]*100 >= (100-tolerance)):
                frame = swapper.get(frame, faces[max_index], src_face[0], paste_back=True)
        if preview == -1:
            video_out.write(frame)
    if preview == -1:
        video_out.release()
        return out_path
    else:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
ins_get_image

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = frame_count/fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return f"Resolution: {width}x{height}\nLength: {length}\nFps: {fps}\nFrames: {frame_count}"
    
def update_slider(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = frame_count/fps
    return gr.update(minimum=0,maximum=frame_count,value=frame_count/2)
                     
def show_preview(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def create_interface():
    title = 'Face Swap UI'
    with gr.Blocks(analytics_enabled=False, title=title) as face_swap_ui:
        with gr.Tab("Swap Face Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label='Input Image (Click to select a face)').style(height=400)
                    with gr.Row():
                        analyze_button = gr.Button("Analyze")
                    with gr.Row():
                        with gr.Column():
                            face_num = gr.Number(label='Recognized Faces')
                            face_index_num = gr.Number(label='Face Index', precision=0)
                        selected_face = gr.Image(label='Face to swap', interactive=False)
                        swap_face = gr.Image(label='Swap Face')
                    swap_button = gr.Button("Swap")
                with gr.Column():
                    image_output = gr.Image(label='Output Image',interactive=False)
                    #text_output = gr.Textbox(placeholder="What is your name?")
            swap_button.click(fn=swap_face_fct, inputs=[image_input, face_index_num, swap_face], outputs=[image_output])
            image_input.select(select_handler, image_input, [selected_face, face_index_num])
            analyze_button.click(fn=get_faces, inputs=image_input, outputs=[image_input,face_num])
        with gr.Tab("Swap Face Video"):
            with gr.Row():
                with gr.Column():
                    source_video = gr.Video()
                    video_info = gr.Textbox(label="Video Information")
                    gr.Markdown("Select a frame for preview with the slider. Then select the face which should be swapped by clicking on it with the cursor")
                    video_position = gr.Slider(label="Frame preview",interactive=True)
                    frame_preview = gr.Image(label="Frame preview")
                    face_index = gr.Textbox(label="Face-Index",interactive=False)
                    with gr.Row():
                        dest_face_vid = gr.Image(Label="Face tow swap",interactive=True)
                        source_face_vid = gr.Image(Label="New Face")
                    gr.Markdown("The higher the tolerance the more likely a wrong face will be swapped. 30-40 is a good starting point.")
                    face_tolerance = gr.Slider(label="Tolerance",value=40,interactive=True)
                    preview_video = gr.Button("Preview")
                    video_file_path = gr.Text(label="Output Video path incl. file.mp4 (when left empty it will be put in the gradio temp dir)")
                    process_video = gr.Button("Process")
                with gr.Column():
                    with gr.Column(scale=1):
                        image_output = gr.Image()
                        output_video = gr.Video(interactive=False)
                    with gr.Column(scale=1):
                        pass
            # Component Events
            source_video.upload(fn=analyze_video,inputs=source_video,outputs=video_info)
            video_info.change(fn=update_slider,inputs=source_video,outputs=video_position)
            #preview_button.click(fn=show_preview,inputs=[source_video, video_position],outputs=frame_preview)
            frame_preview.select(select_handler, frame_preview, [dest_face_vid, face_index ])
            video_position.change(show_preview,inputs=[source_video, video_position],outputs=frame_preview)
            process_video.click(fn=swap_video_fct,inputs=[source_video,video_file_path,source_face_vid,dest_face_vid, face_tolerance], outputs=output_video)
            preview_video.click(fn=swap_video_fct,inputs=[source_video,video_file_path,source_face_vid,dest_face_vid, face_tolerance, video_position], outputs=image_output)
        
    face_swap_ui.queue().launch()
    #face_swap_ui.launch()



if __name__ == "__main__":

    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

    create_interface()  