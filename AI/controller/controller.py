from model.model import Model


class Controller:

    @staticmethod
    def start_app():
        from view.view import App
        app = App()

    @staticmethod
    def run_ai():
        Model.main()