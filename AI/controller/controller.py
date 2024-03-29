class Controller:

    @staticmethod
    def start_app():
        from view.main_view import App
        App()

    @staticmethod
    def run_ai(image_path):
        from model.model import Model
        return Model.main(image_path)
