
from openpyxl import *
from tkinter import *
import tkinter as tk
from tkinter.filedialog import Open
from PIL import Image, ImageTk
from model import load_model
from image_processing import load_image_val
import cv2
import webbrowser
import time
import threading
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import glob
import imageio

# Define function to show frames
def show_frames_input(file_path):
    imgin = cv2.resize(cv2.imread(file_path)[...,::-1],(500,500))
    # Get the latest frame and convert into Image
    img = Image.fromarray(imgin)

    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image = img)
    input_img.imgtk = imgtk
    input_img.configure(image = imgtk)
    
    # Repeat after an intercal to capture continiously
    # input_img.after(10, show_frames_input)

def show_frames_output():
    imgin = cv2.resize(cv2.imread('img/output.png')[...,::-1],(500,500))
    # Get the latest frame and convert into Image
    img = Image.fromarray(imgin)

    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image = img)
    output_img.imgtk = imgtk
    output_img.configure(image = imgtk)
    
    # # Repeat after an intercal to capture continiously
    # output_img.after(20, show_frames_output)

def onOpen():
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
        dlg = Open(filetypes = ftypes)
        fl = dlg.show()
        if fl != '':
            input_img = load_image_val(fl).numpy()
            input_img = input_img.reshape((1,224,224,3))
            output_img = model(input_img, training=True)[0].numpy()
            
            plt.axis('off')
            plt.imshow(output_img)
            plt.savefig('img/output.png',bbox_inches='tight')
            show_frames_input(fl)
            show_frames_output()

def OpenVideo():
        dlg = Open()
        fl = dlg.show()
        if fl != '':
            cap = cv2.VideoCapture(fl)
            i = 0
            # choose codec according to format needed
            w,h = cap.read()[1].shape[1], cap.read()[1].shape[0]
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    cv2.imwrite("input/" + str(i) + '.png', frame)
                    input_img = load_image_val("input/" + str(i) + '.png').numpy()
                    input_img = input_img.reshape((1,224,224,3))
                    output_img = model(input_img, training=True)[0].numpy()
                    
                    plt.imshow(output_img)
                    plt.axis('off')
                    plt.savefig("output/" + str(i) + '.png',bbox_inches='tight')
                    i = i + 1
                if ret == False:
                    break
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter('video/out.mp4', fourcc, 40, (w, h))
        for index in range (i):
           path = 'output/' + str(index) + '.png'
           img = cv2.resize(cv2.imread(path),(w,h))
           video.write(img)
        open_v['text'] = 'Ready'
        cv2.destroyAllWindows()
        video.release()

def stream(label, path):
    video_name = path #This is your video file path
    video = imageio.get_reader(video_name)
    for image in video.iter_data():
        frame_image = ImageTk.PhotoImage(Image.fromarray(image))
        label.config(image=frame_image)
        label.image = frame_image
        time.sleep(0.025)

def show_video_output():
    # t = threading.Thread(target=show_video_output_out)
    # t.start()
    thread = threading.Thread(target=stream, args=(input_img,"video/in.mp4"))
    thread.daemon = 1
    thread.start()

    thread1 = threading.Thread(target=stream, args=(output_img,"video/out.mp4"))
    thread1.daemon = 1
    thread1.start()

# Load Model
model = load_model()

# Driver code
if __name__ == "__main__":
    
    # Create a GUI window
    root = Tk()

    # Set the background colour of GUI window
    root.configure(background = "pink")

    # Set the title of GUI window
    root.title("De Make-Up GUI")

    # Set the configuration of GUI window
    root.geometry("1600x900+0+0")
    root.attributes('-fullscreen', True)

    # Create a Title label
    title = Label(
        root,
        text = "Lipstick Everywhere", 
        font = ("Helvetica", 60, "bold", "italic"),
        bg = "pink",
        fg = '#FF4600')
        
    title.place(
        relx = .5,
        rely = .1,
        anchor = 'center')
    
    i = Label(
        root,
        text = "MAKE UP", 
        font = ("Helvetica", 35, "italic"),
        fg = '#00FF70',
        bg = 'pink')
    
    i.place(
        relx = .3,
        rely = .22,
        anchor = 'center')

    o = Label(
        root,
        text = "WITH OUT MAKE UP", 
        font = ("Helvetica", 35, "italic"),
        fg = '#00FF70',
        bg = 'pink')
    
    o.place(
        relx = .7,
        rely = .22,
        anchor = 'center')
    # Create a Label to capture the Video frames
    input_img = Label(root)
    input_img.place(
        relx = .3,
        rely = .5,
        anchor = 'center')

    output_img = Label(root)
    output_img.place(
        relx = .7,
        rely = .5,
        anchor = 'center')

    # Button Open Img
    open_b = Button(root, text ="Open Image", height = "1", width = "10", background = "#00E4FF", foreground = "#fff",
        font = ("Helvetica", 20, "bold", "italic"), command = onOpen, activebackground='red') 
    open_b.place(
        relx = .3,
        rely = .8,
        anchor = 'center')

    # Button Open vIDEO
    open_b1 = Button(root, text ="Open Video", height = "1", width = "10", background = "#00E4FF", foreground = "#fff",
        font = ("Helvetica", 20, "bold", "italic"), command = OpenVideo, activebackground='red') 
    open_b1.place(
        relx = .7,
        rely = .8,
        anchor = 'center')

    # Show Video
    open_v = Button(root, text ="Not Ready", height = "2", width = "15", background = "#00E4FF",
        font = ("Helvetica", 10, "bold", "italic"), command = show_video_output, activebackground='red') 
    open_v.place(
        relx = .7,
        rely = .9,
        anchor = 'center')

    # Start the GUI
    root.mainloop()
    
print("Okela. Right now!")