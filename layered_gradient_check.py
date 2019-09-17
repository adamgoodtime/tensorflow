from brilliant_BP_formulas import *
import numpy as np


def check_gradient(f, df, x0, tries=1, deltas=(1e-2, 1e-4, 1e-6)):
    # Init around the point x0
    f0 = f(x0)
    g0 = df(x0)

    # For different variations tries if the gradient is well approximated with finite difference
    for k_dx in range(tries):
        dh = 2 * np.random.random((4, 3)) - 1
        do = 2 * np.random.random((3 + 1, 1)) - 1

        approx_err = np.zeros(len(deltas))
        df_h = np.inner(dh, g0[0])
        dh_tot = 0.0
        for rowd, rowg in zip(dh, g0[0]):
            for col in range(len(rowd)):
                dh_tot += rowd[col] * rowg[col]
        df_o = np.inner(do, g0[1])
        do_tot = 0.0
        for rowd, rowg in zip(do, g0[1]):
            for col in range(len(rowd)):
                do_tot += rowd[col] * rowg[col]
        df_g = df_h + df_o
        df_g = dh_tot + do_tot

        for k, d in enumerate(deltas):
            f1 = f([x0[0] + (d * dh), x0[1] + (d * do)])
            df = (f1 - f0) / d

            approx_err[k] = np.log10(np.abs(df_g - df) + 1e-20)

        if (np.diff(approx_err) < -1).all() or (approx_err < -20).all():
            print('Gradient security check OK: the gradient df is well approximated by finite difference.')

        else:
            raise ValueError(
                '''GRADIENT SECURITY CHECK ERROR:
                Detected by approximating the gradient with finite difference.
                The cost function or the gradient are not correctly computed.
                The approximation D(eta) = (f(x + eta dx) - f(x)) / eta should converge toward df=grad*dx.

                Instead \t D({:.3g}) = {} \t df = {:.3g}
                Overall for

                \t \t  eta \t \t \t {}
                log10( |D(eta) - df|) \t {} '''.format(d, df, df_g, deltas, approx_err))


if __name__ == "__main__":
    brilliant = True
    hidden_weights = 2 * np.random.random((4, 3)) - 1
    output_weights = 2 * np.random.random((3 + 1, 1)) - 1
    w0 = [hidden_weights, output_weights]
    f = lambda x: brilliant_BP(x, return_error=True)
    df = lambda x: brilliant_BP(x, return_error=False)

    check_gradient(f, df, w0)

