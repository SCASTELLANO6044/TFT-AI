import customtkinter
import os.path
from PIL import Image


class OutputFrame(customtkinter.CTkFrame):

    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        image_path = os.path.join('.', 'ISIC_0024306.jpg')

        self.my_image = customtkinter.CTkImage(light_image=Image.open(image_path),
                                               dark_image=Image.open(image_path),
                                               size=(400, 300))

        self.image_label = customtkinter.CTkLabel(self, image=self.my_image, text="")
        self.image_label.grid(row=1, column=0, padx=0, pady=10, sticky="ew")