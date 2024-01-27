import customtkinter


class IntputFrame(customtkinter.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.theme_switch = customtkinter.CTkSwitch(self, text="", command=master.change_theme, variable=None,
                                                    onvalue="on",
                                                    offvalue="off")
        self.theme_switch.grid(row=0, column=0, padx=0, pady=10, sticky="ne")

        self.file_entry = customtkinter.CTkButton(self, text="Busca tu imagen.", command=master.browse_file,
                                                  fg_color="#F39A0E", hover_color="#c47a0b", font=("Helvetica", 15),
                                                  text_color="#33383F")
        self.file_entry.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.predict_button = customtkinter.CTkButton(self, text="Calcular predicción.", command=master.predict_button_action,
                                                      fg_color="#0E68A4", hover_color="#0b4c77", font=("Helvetica", 15),
                                                      text_color="#33383F")
        self.predict_button.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
