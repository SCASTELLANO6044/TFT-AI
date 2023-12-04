from model.model import Model


class Controller:

    @staticmethod
    def start_app():
        from view.view import App
        app = App()
        app.mainloop()

    @staticmethod
    def run_ai():
        Model.main()