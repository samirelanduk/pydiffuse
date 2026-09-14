from unittest import TestCase

import torch

from pydiffuse.layers import convolution, gelu, group_norm, layer_norm, linear, silu


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

    def test_linear_layer_no_bias(self):
        input = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
        weights = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        output = linear(weights, None, input)
        self.assertEqual(output.tolist(), [[300, 700, 1100], [700, 1740, 2780]])


class GroupNormLayerTests(TestCase):
    def test_group_norm_layer(self):
        input = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        bias = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        output = group_norm(weights, bias, input)
        self.assertTrue(
            torch.allclose(
                output,
                torch.tensor(
                    [85.3615, 182.4338, 291.2169, 411.7108, 543.9154, 687.8309]
                ),
            )
        )

    def test_group_norm_layer_custom_group_size(self):
        input = torch.tensor([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        bias = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        output = group_norm(weights, bias, input, groups=2)
        self.assertTrue(
            torch.allclose(
                output,
                torch.tensor(
                    [89.3096, 194.6548, 340.0891, 357.2382, 486.6370, 680.1783]
                ),
            )
        )

    def test_group_norm_layer_higher_dimensions(self):
        input = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[2.0, 4.0], [8.0, 16.0]],
                [[32.0, 64.0], [128.0, 256.0]],
                [[1.0, 1.0], [2.0, 2.0]],
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
                        [[84.7248, 89.0891], [93.4535, 97.8178]],
                        [[204.3643, 213.0930], [221.8218, 230.5505]],
                        [[300.0000, 307.2231], [314.4463, 321.6694]],
                        [[332.5839, 351.8457], [390.3691, 467.4160]],
                        [[483.0479, 501.9163], [539.6531, 615.1266]],
                        [[557.7230, 557.7230], [558.4305, 558.4305]],
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
                        [[702.0, 729.0], [810.0, 837.0], [918.0, 945.0]],
                        [[79.0, 82.0], [91.0, 94.0], [103.0, 106.0]],
                        [[64.0, 73.0], [100.0, 109.0], [136.0, 145.0]],
                        [[178.0, 181.0], [190.0, 193.0], [202.0, 205.0]],
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
                    ]
                ),
            )
        )

    def test_convolution_layer_with_padding_at_end(self):
        output = convolution(
            self.weights, self.bias, self.input, padding=1, pad_at_end=True
        )
        self.assertTrue(
            torch.allclose(
                output,
                torch.tensor(
                    [
                        [
                            [702.0, 729.0, 495.0],
                            [810.0, 837.0, 567.0],
                            [918.0, 945.0, 639.0],
                            [648.0, 666.0, 450.0],
                        ],
                        [
                            [79.0, 82.0, 85.0],
                            [91.0, 94.0, 97.0],
                            [103.0, 106.0, 109.0],
                            [115.0, 118.0, 121.0],
                        ],
                        [
                            [64.0, 73.0, 55.0],
                            [100.0, 109.0, 79.0],
                            [136.0, 145.0, 103.0],
                            [106.0, 112.0, 80.0],
                        ],
                        [
                            [178.0, 181.0, 85.0],
                            [190.0, 193.0, 85.0],
                            [202.0, 205.0, 85.0],
                            [85.0, 85.0, 85.0],
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
                        [[702.0], [918.0]],
                        [[79.0], [103.0]],
                        [[64.0], [136.0]],
                        [[178.0], [202.0]],
                    ]
                ),
            )
        )


class SiluLayerTests(TestCase):
    def test_silu_layer(self):
        input = torch.tensor(
            [[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], [-1.5, -0.5, 0.5, 3.0, 6.0, 12.0]]
        )
        output = silu(input)
        self.assertTrue(
            torch.allclose(
                output,
                torch.tensor(
                    [
                        [-0.142278, -0.238406, -0.268941, 0.0, 0.731059, 1.761594],
                        [-0.273638, -0.188770, 0.311230, 2.857722, 5.985165, 11.999926],
                    ]
                ),
            )
        )


class GeluLayerTests(TestCase):
    def test_gelu_layer(self):
        input = torch.tensor(
            [[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], [-1.5, -0.5, 0.5, 3.0, 6.0, 12.0]]
        )
        output = gelu(input)
        self.assertTrue(
            torch.allclose(
                output,
                torch.tensor(
                    [
                        [-0.004050, -0.045500, -0.158655, 0.0, 0.841345, 1.954500],
                        [-0.100211, -0.154269, 0.345731, 2.995950, 6.0, 12.0],
                    ]
                ),
                atol=1e-6,
            )
        )
