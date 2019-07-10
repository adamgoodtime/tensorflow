import numpy as np
from graz_non_tensor import *

# target hz - z * pseudo_Dev * pre_synap spike + c + (1 - alpha)

def check_gradient(f, df, x0, tries=10, deltas=(1, 1e-2, 1e-4, 1e-6)):
    # Init around the point x0
    f0 = f(x0)
    g0 = df(x0)
    g0 = np.array(g0)
    g0 = g0.reshape(g0.size)

    # For different variations tries if the gradient is well approximated with finite difference
    for k_dx in range(tries):
        # random steps
        dx = np.random.randn(x0.size) + mean
        dx *= I
        # dx = np.random.randn(len(x0), len(x0[0]))

        # initialise error
        approx_err = np.zeros(len(deltas))
        # find df at random point dx
        df_g = np.inner(dx, g0)
        df_0 = np.inner(x0, g0)

        # loop through the different deltas
        for k, d in enumerate(deltas):
            # calculate the value at the next point
            print "\nx0:", x0
            print "d * dx:", d * dx
            print "all:", x0 + d * dx
            print "df_0:", df_0
            print "diff_0", np.inner(dx - x0, g0)
            f1 = f(x0 + d * dx)
            # approximate the change in value between the 2 points
            df = (f1 - f0) / d
            print "f1:", f1
            print "df", df

            # compare the approximate change in df with actual change
            approx_err[k] = np.log10(np.abs(df_g - df) + 1e-20)

        # if difference is small enough give all clear
        if (np.diff(approx_err) < -1).all() or (approx_err <= -20).all():
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


if __name__ == "__main__":
    # np.random.seed(272727)
    mean = 0
    w0 = np.random.randn(4) + mean
    I = np.array([[0, 1], [0, 0]])
    I = np.array([0, 1, 0, 0])
    w0 *= I
    # f = lambda w: 0.5 * np.sum(w ** 2)
    # df = lambda w: w
    # df = lambda x: f(x) * (1 - f(x))
    # f = lambda x: np.sum(1 / (1 + np.exp(-x)))
    # df = lambda x: np.exp(-x) / ((1 + np.exp(-x))**2)
    f = lambda x: error_and_BP_gradients(x.reshape((int(np.sqrt(len(x))), int(np.sqrt(len(x))))), return_error=True, quadratic=True)
    df = lambda x: error_and_BP_gradients(x.reshape((int(np.sqrt(len(x))), int(np.sqrt(len(x))))), return_error=False)
    check_gradient(f, df, w0)
