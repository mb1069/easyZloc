from model.model import load_new_model


def test_load_new_model_should_compile():
    model = load_new_model(31)
    model.compile()
    print(model.summary())

