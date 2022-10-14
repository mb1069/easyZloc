from model.model import load_new_model


def test_load_new_model_should_compile():
    model = load_new_model(31, lr=0.01)
    model.compile()
    print(model.summary())

