import customtkinter
from controller.controller import Controller
from view.input_view import IntputFrame
from view.output_view import OutputFrame


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

    def predict_button_action(self):
        prediction = Controller.run_ai()
        print(prediction)

    def change_theme(self):
        if customtkinter.get_appearance_mode() == "Dark":
            customtkinter.set_appearance_mode("light")
        else:
            customtkinter.set_appearance_mode("dark")

    def set_image_label(self, content):
        self.image_label.configure(self, text=content)

app = App()
app.mainloop()
