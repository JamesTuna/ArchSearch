from NASsearch.wl_kernel import WLKernel

NASNet = dict(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = dict(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

Test = dict(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
    ],
    normal_concat=[2],
    reduce=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[2]
)

Test2 = dict(
    normal=[
        ('max_pool_3x3', 1),
        ('avg_pool_3x3', 0),
    ],
    normal_concat=[2],
    reduce=[
        ('max_pool_3x3', 1),
        ('avg_pool_3x3', 0),
    ],
    reduce_concat=[2]
)


def run():
    kernel = WLKernel(
        N=3,
        net1=Test2,
        net2=Test,
        level=2,
        # B1=5,
        # B2=2,
    )
    print(kernel.run())


if __name__ == "__main__":
    run()
