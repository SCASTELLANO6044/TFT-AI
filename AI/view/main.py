import customtkinter


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("TFT-AI")
        self.geometry("600x400")
        self.grid_columnconfigure(0, weight=1)

        self.predict_button = customtkinter.CTkButton(self, text="Predict", command=self.predict_button_action)
        self.predict_button.grid(row=1, column=0, padx=20, pady=20, sticky="ew", columnspan=2)
        self.file_entry = customtkinter.CTkEntry(self)
        self.file_entry.grid(row=0, column=0, padx=20, pady=(0, 20), sticky="ew", columnspan = 2)

    def predict_button_action(self):
        print("button pressed")


app = App()
app.mainloop()
