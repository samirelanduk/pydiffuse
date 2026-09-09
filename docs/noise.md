# Noise

'Noise' is distortion added to an image.

## Images

An image, here, is a 3D tensor of width, height, and ‘channels’.
In regular pixel images there are three channels representing RGB values for the pixel at that position.
For images in latent space there will be more (typically four) and they don’t correspond 1:1 to anything in the original image.

The images will have a mean value (the average of all the numbers in the tensor) and a variance (the average squared distance of each value from the mean).
In pixel images these can vary - the mean will be somewhere between 0 and 255, and the variance will be roughly on that order of magnitude.
For latent images though, the encoding process typically ensures they have a mean of roughly zero, and a variance of one.
The information is in the relationship between values, the mean and variance are assumed to be held around these fixed values.

## Noise

To add ‘noise’ to an image is to make small random adjustments to these values to destroy some (though usually not all) of the information.
If these random numbers are large compared with the scale of the numbers in your image, you are going to essentially completely destroy the original image.
If they are very small compared with the scale of the image, the perturbations are going to be insignificant.

In the case of latent images, you first create a new tensor of the same shape, but where every value is random - from a standard normal distribution so it has the same mean zero, variance one properties as the image.
This is essentially the ‘pure noise’ equivalent of the image.

You then decide what split of original-to-noise you want.
This can be of any ratio - 0.99 original data vs 0.01 noise, 0.5 each, 0.1 original data and 0.9 noise etc.

The most straightforward way to do this would be to multiply each tensor (the original and the full random noise) by that fraction, and then add them.
For example if you wanted 0.25 image vs 0.75 noise, you multiply the image tensor by 0.25, the noise tensor by 0.75, and add them.
This would produce a tensor with the correct amount of noise, and it would still have a mean of zero because both summed tensors had a mean of zero.

However, the variance would be wrong.
If you make the values 4x smaller, the variance ends up 16x smaller because the variance is the square of the scaling factor.
So the variance of the image tensor is 1/16, the variance of the noise tensor is now 9/16, and the variance of the final noised tensor is 10/16 - 5/8 when it should be 1.

So for a 0.25/0.75 split, you don’t scale by these fractions, you scale by the square root of them.
So you scale the image by 0.5, to produce a tensor with a variance of 0.
25, and you scale the noise by 0.84, so its variance becomes 0.75.
Then you add them.

## Sigma

For any noising, there is a noise level and a corresponding 'original' level, which sum to one.
The square roots of each are the two scaling factors.
It is the relative size of these two scaling factors which determines how noisy the image is, not their absolute values.
You could double both and the resulting image would be just as noisy, it would just have a higher variance (specifically four times higher).

This ratio of the two scaling factors is called sigma, and is an alternate way of describing the noise applied.
A sigma of zero means no noise is added, a sigma of one means equal noise and image (noise factor of 0.5), and as noise factor approaches one, sigma goes off to infinity.

Rather than noising by scaling both tensors by the square roots, you can just add the noise tensor multiplied by sigma, and then divide by the appropriate factor to reduce the variance down to one.
It is mathematically identical to the previous method.