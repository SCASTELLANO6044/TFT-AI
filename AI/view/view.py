import customtkinter
from controller.controller import Controller


class OutputFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)


def predict_button_action():
    Controller.run_ai()


class IntputFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.button = customtkinter.CTkButton(self, text="Predict", command=predict_button_action)
        self.button.grid(row=2, column=0, padx=10, pady=10, sticky="sew")


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
