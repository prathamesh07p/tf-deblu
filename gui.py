import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image, ImageTk

class ImageDeblurApp(tk.Tk):
    def __init__(self):
        """
        Initializes the GUI window for the Image Deblur application.
        """
        super().__init__()
        self.title("Image Deblur")
        self.geometry("500x450")
        self.configure(background="white")

        self.label_input = tk.Label(self, text="Click to see the Input Image", fg="black", bg="white")
        self.button_input = tk.Button(self, text="Input Image", width=20, command=self.input_image)
        self.label_deblur = tk.Label(self, text="Click to perform the Deblur process", fg="black", bg="white")
        self.button_deblur = tk.Button(self, text="Deblur", width=20, command=self.run)
        self.label_output = tk.Label(self, text="Click to see the Output Image", fg="black", bg="white")
        self.button_output = tk.Button(self, text="Output Image", width=20, command=self.output_image)
        self.button_exit = tk.Button(self, text="Exit", width=20, command=self.destroy)

        self.label_input.grid(row=0, column=0, sticky="w")
        self.button_input.grid(row=0, column=1)
        self.label_deblur.grid(row=1, column=0, sticky="w")
        self.button_deblur.grid(row=1, column=1)
        self.label_output.grid(row=2, column=0, sticky="w")
        self.button_output.grid(row=2, column=1)
        self.button_exit.grid(row=3, column=1)

    def input_image(self):
        """
        Method to input an image and show it using the 'show_image' method.
        """
        self.show_image("eg.png", "Before Deblur")

    def run(self):
        """
        This function runs the specified command to execute the 'test.py' file.
        """
        os.system('python test.py')

    def output_image(self):
        """
        Method to output an image, showing the image after deblur if available, otherwise prints an error message.
        No parameters or return types.
        """
        try:
            self.show_image("eg_deblur.png", "After Deblur")
        except FileNotFoundError:
            print("Please run the Deblur process first")

    def show_image(self, image_path, label_text):
        """
        Show an image with a label in the GUI.

        Args:
            image_path (str): The file path of the image.
            label_text (str): The text to be displayed as the label.

        Returns:
            None
        """
        image = Image.open(image_path)
        label = tk.Label(self, text=label_text, fg="black", bg="white")
        label.grid(row=4, column=0, sticky="w")
        tkimage = ImageTk.PhotoImage(image)
        image_label = tk.Label(self, image=tkimage)
        image_label.image = tkimage
        image_label.grid(row=4, column=1)

if __name__ == "__main__":
    app = ImageDeblurApp()
    app.mainloop()
