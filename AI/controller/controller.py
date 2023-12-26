from model.model import Model


class Controller:

    @staticmethod
    def start_app():
        from view.view import App
        App()

    @staticmethod
    def run_ai():
        return Model.main()