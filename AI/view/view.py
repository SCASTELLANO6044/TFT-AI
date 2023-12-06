import os.path

import customtkinter
from controller.controller import Controller
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

    def set_image_label(self, content):
        self.image_label.configure(self, text=content)


def predict_button_action():
    prediction = Controller.run_ai()



def change_theme():
    if customtkinter.get_appearance_mode() == "Dark":
        customtkinter.set_appearance_mode("light")
    else:
        customtkinter.set_appearance_mode("dark")


class IntputFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.theme_switch = customtkinter.CTkSwitch(self, text="", command=change_theme, variable=None, onvalue="on",
                                                    offvalue="off")
        self.theme_switch.grid(row=0, column=0, padx=0, pady=10, sticky="ne")

        self.file_entry = customtkinter.CTkEntry(self, placeholder_text="please enter an image")
        self.file_entry.grid(row=1, column=0, padx=50, pady=10, sticky="sew")

        self.predict_button = customtkinter.CTkButton(self, text="Predict", command=predict_button_action)
        self.predict_button.grid(row=2, column=0, padx=10, pady=10, sticky="sew")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("TFT-AI")
        self.geometry("1000x500")
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.output_frame = OutputFrame(self)
        self.output_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        self.output_frame.configure(fg_color="transparent")

        self.input_frame = IntputFrame(self)
        self.input_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")


app = App()
app.mainloop()
