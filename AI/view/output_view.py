import customtkinter
import os.path
from PIL import Image


class OutputFrame(customtkinter.CTkFrame):

    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.image_path = os.path.join('.', 'media', 'ulpgc-logo.png')

        self.my_image = customtkinter.CTkImage(light_image=Image.open(self.image_path),
                                               size=(400, 210))

        self.image_label = customtkinter.CTkLabel(self, image=self.my_image, text="")
        self.image_label.grid(row=1, column=0, padx=0, pady=10, sticky="ew")

    def display_image(self, image_path):
        self.image_path = image_path
        self.my_image.configure(light_image=Image.open(self.image_path),
                                size=(400, 300))

    def get_image(self):
        return self.image_path

    def set_image_label(self, prediction):
        self.image_label.configure(text=prediction)