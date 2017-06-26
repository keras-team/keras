def update(x, new_x):
    return x, new_x


def update_add(x, increment):
    return x, x + increment


def update_sub(x, decrement):
    return x, x - increment


def moving_average_update(x, value, momentum):
    return x, x * momentum + value * (1.0 - momentum)
