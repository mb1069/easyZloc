from model.model import load_new_model, save_model, load_trained_model

def test_save_and_load_model():
    mname='test'
    model = load_new_model(31, lr=0.01)
    model.compile()
    save_model(model, mname)
    model = load_trained_model(mname)

def test_load_new_model_should_compile():
    model = load_new_model(31, lr=0.01)
    model.compile()
    print(model.summary())

