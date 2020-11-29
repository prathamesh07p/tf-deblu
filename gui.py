from tkinter import *
from tkinter import filedialog
import os
import sys
from PIL import Image, ImageTk

def inputImage():
    label_before = Label(window,
                         text = "Before Deblur",
                         fg = "black",
                         bg = "white")
    label_before.grid(row = 4 ,column = 0)
    im = Image.open("eg.png")
    tkimage = ImageTk.PhotoImage(im)
    myvar=Label(window,image = tkimage)
    myvar.image = tkimage
    myvar.grid(row = 4 ,column = 1)

def run():
    os.system('python test.py')

def outputImage():
    try:
        im = Image.open('eg_deblur.png')
        label_after = Label(window,
                            text = "After Deblur",
                            fg = "black",
                            bg = "white")
        label_after.grid(row = 5 ,column = 0)
        tkimage = ImageTk.PhotoImage(im)
        myvar=Label(window,image = tkimage)
        myvar.image = tkimage
        myvar.grid(row = 5 ,column = 1)
    except:
        print("Please run the Deblur process first")

window = Tk()
window.title('Image Deblurring')
window.geometry("500x450")
window.config(background = "white")
label_input = Label(window,
                    text = "Click to see the Input Image",
                    fg = "black",
                    bg = "white",)
button_input = Button(window,
                        text = "Input Image",
                        width = 20,
                        command = inputImage)
label_deblur = Label(window,
                     text = "Click to perform the Deblur process",
                     fg = "black",
                     bg = "white")
button_deblur = Button(window,
                     text = "Deblur",
                     width = 20,
                     command = run)
label_output = Label(window,
                            text = "Click to see the Output Image",
                            fg = "black",
                            bg = "white")
button_output = Button(window,
                       text = "Output Image",
                       width = 20,
                       command = outputImage)
button_exit = Button(window,
                     text = "Exit",
                     width = 20,
                     command = exit)
label_team = Label(window,
                      text = "Guided by Sonali Sawant Ma'am\nBy TE-B B2 2020 Batch\nFrom TEETB219 to TEETB336 ,TEETB273 and TEETB274",
                      fg = "black",
                      bg = "white")

label_input.grid(row = 0 ,column = 0 ,sticky = W)
button_input.grid(row = 0 ,column = 1)
label_deblur.grid(row = 1 ,column = 0 ,sticky = W)
button_deblur.grid(row = 1 ,column = 1)
label_output.grid(row = 2 ,column = 0 ,sticky = W)
button_output.grid(row = 2 ,column = 1)
button_exit.grid(row = 3 ,column = 1)
label_team.grid(row = 6 ,column = 0)
window.mainloop()
