import customtkinter
from controller.controller import Controller


def predict_button_action():
    Controller.run_ai()


class InputFrame(customtkinter.CTkFrame):
    def __int__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, 1)

        checkbox = customtkinter.CTkCheckBox(self, text="test")
        checkbox.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="w")
        predict_button = customtkinter.CTkButton(self, text="Predict", command=predict_button_action)
        predict_button.grid(row=1, column=1, padx=20, pady=20, sticky="ew")


class OutputFrame(customtkinter.CTkFrame):
    def __int__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, 1)

        checkbox = customtkinter.CTkCheckBox(self, text="test")
        checkbox.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="w")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("TFT-AI")
        self.geometry("1000x500")
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.input_frame = InputFrame(self)
        self.input_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsew")
        self.input_frame.configure(fg_color="transparent")
        self.output_frame = OutputFrame(self)
        self.output_frame.grid(row=0, column=1, padx=(0, 10), pady=(10, 0), sticky="nsew")


app = App()
app.mainloop()
