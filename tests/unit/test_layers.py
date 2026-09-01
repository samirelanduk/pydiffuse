from unittest import TestCase

import torch

from pydiffuse.layers import convolution, group_norm, layer_norm, linear


class LinearLayerTests(TestCase):
    def test_linear_layer_vector_input(self):
        input = torch.tensor([10, 20, 30, 40])
        weights = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        bias = torch.tensor([12, 13, 14])
        output = linear(weights, bias, input)
        self.assertEqual(output.tolist(), [312, 713, 1114])

    def test_linear_layer_matrix_input(self):
        input = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
        weights = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        bias = torch.tensor([12, 13, 14])
        output = linear(weights, bias, input)
        self.assertEqual(output.tolist(), [[312, 713, 1114], [712, 1753, 2794]])


class GroupNormLayerTests(TestCase):
    def test_group_norm_layer(self):
        input = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]]
        )
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        bias = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        output = group_norm(weights, bias, input)
        self.assertTrue(
            torch.allclose(
                output,
                torch.tensor(
                    [
                        [85.3615, 182.4338, 291.2169, 411.7108, 543.9154, 687.8309],
                        [91.2266, 184.3003, 281.9915, 390.7649, 525.3966, 719.1333],
                    ]
                ),
            )
        )

    def test_group_norm_layer_custom_group_size(self):
        input = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]]
        )
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        bias = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        output = group_norm(weights, bias, input, groups=2)
        self.assertTrue(
            torch.allclose(
                output,
                torch.tensor(
                    [
                        [87.7526, 200.0000, 336.7421, 351.0106, 500.0000, 673.4841],
                        [89.3096, 194.6548, 340.0891, 357.2382, 486.6370, 680.1783],
                    ]
                ),
            )
        )

    def test_group_norm_layer_higher_dimensions(self):
        input = torch.tensor(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[9.0, 10.0], [11.0, 12.0]],
                    [[2.0, 4.0], [8.0, 16.0]],
                    [[32.0, 64.0], [128.0, 256.0]],
                    [[1.0, 1.0], [2.0, 2.0]],
                ]
            ]
        )
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        bias = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        output = group_norm(weights, bias, input, groups=3)
        self.assertTrue(
            torch.allclose(
                output,
                torch.tensor(
                    [
                        [
                            [[84.7248, 89.0891], [93.4535, 97.8178]],
                            [[204.3643, 213.0930], [221.8218, 230.5505]],
                            [[300.0000, 307.2231], [314.4463, 321.6694]],
                            [[332.5839, 351.8457], [390.3691, 467.4160]],
                            [[483.0479, 501.9163], [539.6531, 615.1266]],
                            [[557.7230, 557.7230], [558.4305, 558.4305]],
                        ]
                    ]
                ),
            )
        )


class LayerNormLayerTests(TestCase):
    def test_layer_norm_layer(self):
        input = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]]
        )
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        bias = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        output = layer_norm(weights, bias, input)
        self.assertTrue(
            torch.allclose(
                output,
                torch.tensor(
                    [
                        [85.3615, 182.4338, 291.2169, 411.7108, 543.9154, 687.8309],
                        [91.2266, 184.3003, 281.9915, 390.7649, 525.3966, 719.1333],
                    ]
                ),
            )
        )

    def test_layer_norm_layer_sequence_input(self):
        input = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[2.0, 4.0, 8.0, 16.0], [1.0, 1.0, 1.0, 1.0]],
            ]
        )
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0])
        bias = torch.tensor([100.0, 200.0, 300.0, 400.0])
        output = layer_norm(weights, bias, input)
        self.assertTrue(
            torch.allclose(
                output,
                torch.tensor(
                    [
                        [
                            [86.5836, 191.0558, 313.4164, 453.6654],
                            [86.5836, 191.0558, 313.4164, 453.6654],
                        ],
                        [
                            [89.7424, 186.9449, 302.7975, 463.4103],
                            [100.0000, 200.0000, 300.0000, 400.0000],
                        ],
                    ]
                ),
            )
        )


class ConvolutionLayerTests(TestCase):
    def setUp(self):
        self.input = torch.tensor(
            [
                [
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0],
                        [17.0, 18.0, 19.0, 20.0],
                    ],
                    [
                        [21.0, 22.0, 23.0, 24.0],
                        [25.0, 26.0, 27.0, 28.0],
                        [29.0, 30.0, 31.0, 32.0],
                        [33.0, 34.0, 35.0, 36.0],
                        [37.0, 38.0, 39.0, 40.0],
                    ],
                    [
                        [41.0, 42.0, 43.0, 44.0],
                        [45.0, 46.0, 47.0, 48.0],
                        [49.0, 50.0, 51.0, 52.0],
                        [53.0, 54.0, 55.0, 56.0],
                        [57.0, 58.0, 59.0, 60.0],
                    ],
                ],
                [
                    [
                        [61.0, 62.0, 63.0, 64.0],
                        [65.0, 66.0, 67.0, 68.0],
                        [69.0, 70.0, 71.0, 72.0],
                        [73.0, 74.0, 75.0, 76.0],
                        [77.0, 78.0, 79.0, 80.0],
                    ],
                    [
                        [81.0, 82.0, 83.0, 84.0],
                        [85.0, 86.0, 87.0, 88.0],
                        [89.0, 90.0, 91.0, 92.0],
                        [93.0, 94.0, 95.0, 96.0],
                        [97.0, 98.0, 99.0, 100.0],
                    ],
                    [
                        [101.0, 102.0, 103.0, 104.0],
                        [105.0, 106.0, 107.0, 108.0],
                        [109.0, 110.0, 111.0, 112.0],
                        [113.0, 114.0, 115.0, 116.0],
                        [117.0, 118.0, 119.0, 120.0],
                    ],
                ],
            ]
        )
        ones = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        zeros = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        centre = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        diagonal = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        self.weights = torch.tensor(
            [
                [ones, ones, ones],  # sum of the whole window
                [centre, centre, centre],  # sum of the three centre pixels
                [ones, zeros, zeros],  # sum of the red window only
                [diagonal, diagonal, diagonal],  # corners minus the centre
            ]
        )
        self.bias = torch.tensor([0.0, 1.0, 10.0, 100.0])

    def test_convolution_layer(self):
        output = convolution(self.weights, self.bias, self.input)
        self.assertTrue(
            torch.allclose(
                output,
                torch.tensor(
                    [
                        [
                            [[702.0, 729.0], [810.0, 837.0], [918.0, 945.0]],
                            [[79.0, 82.0], [91.0, 94.0], [103.0, 106.0]],
                            [[64.0, 73.0], [100.0, 109.0], [136.0, 145.0]],
                            [[178.0, 181.0], [190.0, 193.0], [202.0, 205.0]],
                        ],
                        [
                            [[2322.0, 2349.0], [2430.0, 2457.0], [2538.0, 2565.0]],
                            [[259.0, 262.0], [271.0, 274.0], [283.0, 286.0]],
                            [[604.0, 613.0], [640.0, 649.0], [676.0, 685.0]],
                            [[358.0, 361.0], [370.0, 373.0], [382.0, 385.0]],
                        ],
                    ]
                ),
            )
        )

    def test_convolution_layer_with_padding(self):
        output = convolution(self.weights, self.bias, self.input, padding=1)
        self.assertTrue(
            torch.allclose(
                output,
                torch.tensor(
                    [
                        [
                            [
                                [282.0, 432.0, 450.0, 306.0],
                                [459.0, 702.0, 729.0, 495.0],
                                [531.0, 810.0, 837.0, 567.0],
                                [603.0, 918.0, 945.0, 639.0],
                                [426.0, 648.0, 666.0, 450.0],
                            ],
                            [
                                [64.0, 67.0, 70.0, 73.0],
                                [76.0, 79.0, 82.0, 85.0],
                                [88.0, 91.0, 94.0, 97.0],
                                [100.0, 103.0, 106.0, 109.0],
                                [112.0, 115.0, 118.0, 121.0],
                            ],
                            [
                                [24.0, 34.0, 40.0, 32.0],
                                [43.0, 64.0, 73.0, 55.0],
                                [67.0, 100.0, 109.0, 79.0],
                                [91.0, 136.0, 145.0, 103.0],
                                [72.0, 106.0, 112.0, 80.0],
                            ],
                            [
                                [115.0, 115.0, 115.0, 28.0],
                                [115.0, 178.0, 181.0, 85.0],
                                [115.0, 190.0, 193.0, 85.0],
                                [115.0, 202.0, 205.0, 85.0],
                                [-11.0, 85.0, 85.0, 85.0],
                            ],
                        ],
                        [
                            [
                                [1002.0, 1512.0, 1530.0, 1026.0],
                                [1539.0, 2322.0, 2349.0, 1575.0],
                                [1611.0, 2430.0, 2457.0, 1647.0],
                                [1683.0, 2538.0, 2565.0, 1719.0],
                                [1146.0, 1728.0, 1746.0, 1170.0],
                            ],
                            [
                                [244.0, 247.0, 250.0, 253.0],
                                [256.0, 259.0, 262.0, 265.0],
                                [268.0, 271.0, 274.0, 277.0],
                                [280.0, 283.0, 286.0, 289.0],
                                [292.0, 295.0, 298.0, 301.0],
                            ],
                            [
                                [264.0, 394.0, 400.0, 272.0],
                                [403.0, 604.0, 613.0, 415.0],
                                [427.0, 640.0, 649.0, 439.0],
                                [451.0, 676.0, 685.0, 463.0],
                                [312.0, 466.0, 472.0, 320.0],
                            ],
                            [
                                [115.0, 115.0, 115.0, -152.0],
                                [115.0, 358.0, 361.0, 85.0],
                                [115.0, 370.0, 373.0, 85.0],
                                [115.0, 382.0, 385.0, 85.0],
                                [-191.0, 85.0, 85.0, 85.0],
                            ],
                        ],
                    ]
                ),
            )
        )

    def test_convolution_layer_with_stride(self):
        output = convolution(self.weights, self.bias, self.input, stride=2)
        self.assertTrue(
            torch.allclose(
                output,
                torch.tensor(
                    [
                        [
                            [[702.0], [918.0]],
                            [[79.0], [103.0]],
                            [[64.0], [136.0]],
                            [[178.0], [202.0]],
                        ],
                        [
                            [[2322.0], [2538.0]],
                            [[259.0], [283.0]],
                            [[604.0], [676.0]],
                            [[358.0], [382.0]],
                        ],
                    ]
                ),
            )
        )
