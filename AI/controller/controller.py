from model.model import Model


class Controller:

    @staticmethod
    def start_app():
        from view.main_view import App
        app = App()
        app.mainloop()

    @staticmethod
    def run_ai(image_path):
        return Model.main(image_path)
