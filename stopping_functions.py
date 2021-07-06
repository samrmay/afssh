def basic_1d(left, right):
    def f(fssh):
        if fssh.r[0] > right and fssh.v[0] >= 0:
            return True
        elif fssh.r[0] < left and fssh.v[0] <= 0:
            return True
        return False

    return f


def reached_ground(fssh):
    if fssh.lam == 0:
        return True
    return False


def no_stop(fssh):
    return False
