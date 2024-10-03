import cv2
import numpy as np
import torch
from tkinter import Tk, Label
from PIL import Image, ImageTk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

window = Tk()
window.title('Monodepth')

frame_label = Label(window)
frame_label.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

processed_label = Label(window)
processed_label.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)

# Load Midas model
model_type = "MiDaS_small"  # Midas 2.1 -Small   [higher inference speed]
# model_type = "DPT_large"    # Midas v3 -Large    [higher depth accuracy]

# Load to GPU if available
model = torch.hub.load('intel-isl/MiDaS', model_type)
model.to(device)
model.eval()

# Load transforms to normalize frame
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

if model_type == 'MiDaS_small':
    transform = midas_transforms.small_transform
elif model_type == 'DPT_large':
    transform = midas_transforms.dpt_transform

# Frame capturing
vid = cv2.VideoCapture(0)

def video_stream():
    ret, frame = vid.read()
    if not ret:
        print("Failed to grab frame")
        window.after(10, video_stream)
        return

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #image=frame
    # Apply frame to transform and move to device
    input_batch = transform(image).to(device)
    
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()
    
    relative_depth = prediction.cpu().numpy()
    relative_depth = cv2.normalize(relative_depth, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    
    # Update original frame
    img = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=img)
    frame_label.imgtk = imgtk
    frame_label.configure(image=imgtk)
    
    # Update depth map
    relative_depth = (relative_depth * 255).astype(np.uint8)
    relative_depth = cv2.applyColorMap(relative_depth, cv2.COLORMAP_INFERNO)

    depth = Image.fromarray(cv2.cvtColor(relative_depth,cv2.COLOR_BGR2RGB))
    depthtk = ImageTk.PhotoImage(image=depth)
    processed_label.imgtk = depthtk
    processed_label.configure(image=depthtk)
    
    window.after(10, video_stream)

video_stream()  
window.mainloop()
vid.release()




