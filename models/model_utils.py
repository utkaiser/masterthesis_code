def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'saved_model.th'))

def load_model(model_type):
    from torch import load
    from os import path
    r = model_type
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'saved_model.th'), map_location='cpu'))
    return r