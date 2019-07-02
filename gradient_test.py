import numpy as np


def check_gradient(f, df, x0, tries=10, deltas=(1e-3, 1e-3, 1e-3)):
    # Init around the point x0
    f0 = f(x0)
    g0 = df(x0)

    # For different variations tries if the gradient is well approximated with finite difference
    for k_dx in range(tries):
        # random steps
        dx = np.random.randn(x0[0].__len__(), x0.__len__())
        dx = np.random.choice(a=[False, True], size=(x0[0].__len__(), x0.__len__()))#, p=[p, 1-p])

        # new_x0 = []
        # for value in x0:
        #     new_x0.append(value.append(np.random.choice(a=[True, True])))
        x0[0].append(True)
        x0[1].append(True)

        # initialise error
        approx_err = np.zeros(len(deltas))
        # find df at random points dx
        df_g = np.inner(dx, g0)
        df_g_x0 = df(x0)
        df_g = sum(sum(np.array(df_g_x0)))

        # loop through the different deltas
        for k, d in enumerate(deltas):
            # calculate the value at the next point
            # f1 = f(x0 + d * dx)
            f1 = f(x0)
            # approximate the change in value between the 2 points
            df = (f1 - f0) / d

            # compare the approximate change in df with actual change
            approx_err[k] = np.log10(np.abs(df_g - df) + 1e-20)

        # if difference is small enough give all clear
        if (np.diff(approx_err) < -1).all() or (approx_err < -20).all():
            print('Gradient security check OK: the gradient df is well approximated by finite difference.')

        else:
            raise ValueError(
                '''GRADIENT SECURITY CHECK ERROR:
                Detected by approximating the gradient with finite difference.
                The cost function or the gradient are not correctly computed.
                The approximation D(eta) = (f(x + eta dx) - f(x)) / eta should converge toward df=grad*dx.

                Instead \t D({:.3g}) = {:.3g} \t df = {:.3g}
                Overall for

                \t \t  eta \t \t \t {}
                log10( |D(eta) - df|) \t {} '''.format(d, df, df_g, deltas, approx_err))

    return "done"


if __name__ == "__main__":
    w0 = np.random.randn(3)
    # f = lambda w: 0.5 * np.sum(w ** 2)
    # df = lambda w: w
    f = lambda x: np.sum(1 / (1 + np.exp(-x)))
    df = lambda x: np.exp(-x) / ((1 + np.exp(-x))**2)
    # df = lambda x: f(x) * (1 - f(x))
    check_gradient(f, df, w0)
