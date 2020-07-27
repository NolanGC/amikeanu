from commons import get_model, get_tensor


def getPred(tensor):
    model = get_model()
    print(model)
    output = model(tensor)
    _, pred = output
    return pred
