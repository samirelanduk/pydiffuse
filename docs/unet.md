# UNets

A UNet is a neural network architecture, used here to predict denoised images from noised images.

## Inputs

In latent diffusion, you provide a UNet with (1) a latent image tensor with some amount of noise applied, and (2) a number indicating how much noise has been applied, and train it to predict the original image.
Usually a third input is provided too, a CLIP tensor representing a text description of the image (the conditioning).

The timestep is converted into a vector using sine and cosine waves, and then passed through a simple learned layer to produce a final vector representation.

The CLIP tensor is provided in chunk form from the CLIP encoder, but the chunks are merged before passing them here.

## Noise Schedule

You provide images at various levels of noising so that the model can deal with very noisy images and barely noisy images.
To do this you have to decide how many noise levels you want, and what noise fractions each should have.

The original latent diffusion model had 1000 such 'timesteps' - 0 to 999, with this number referred to as t (timestep).
Each number is associated with a particular noise level.

The most straightforward way to map t to noise levels would be to just divide by 1000.
So level 0 would be 0.001 noise to 0.999 image, level 1 would be 0.002 noise to 0.998 image, and so on down to 0.999 noise and 0.001 image.

Unfortunately this linear noise scale isn’t very useful, because the initial stages involve quite drastic changes to the ratio of noise to original (doubling over the first step) and then at the noisy end, the amount of noise as a proportion of the previous level of noise is increasing by tiny amounts, so hundreds of timesteps are wasted on imperceptibly different levels of noise.

Instead t is first converted to a number called beta.
The beta at the start and end is defined as two small numbers (0.00085 and 0.012), and the beta values in between are interpolated through their square roots being linearly spaced.
So to work out the beta value at any timestep, you work out which number is that distance between sqrt(0.00085) and sqrt(0.012) and then square that number.
This beta becomes the amount that the noise level increases by at each additional timestep.

To get the noise fraction for a timestep, you subtract beta from one (producing alpha), and then take the cumulative product of all alpha values up to that point, to get ‘alpha_bar’ - the image fraction that you then square root to get the scaling factor.
The idea is that each noise level should be a fixed rate of increase from the previous level.

The noised image is then passed to the UNet, with the t used to generate the noise fraction (the t integer turned into a vector via a sine wave process).
It will output a prediction of the noise that has been applied, and by extension the original image (which is obtained by subtracting this noise prediction from the input image).


## UNet Structure

The network halves the height and width of the tensor several times, then doubles it back up to the size it started at.
Halving loses fine detail, so the tensor at each step on the way down is also passed straight across to the matching step on the way up - these are the skip connections.

The two halves are the strokes of a 'U', and the skip connections run across the middle.
The descending half is the input blocks, the lowest point is the middle block, and the ascending half is the output blocks.

### Input Blocks

The input blocks start with a convolution that gives the latent many more channels, and then alternate between three kinds of layer:

1. Residual blocks, which refine the tensor and are where the timestep is applied.
2. Transformers, which use attention to let parts of the image inform each other, and are where the conditioning is applied.
3. Downsampling convolutions, which halve the height and width.

The output of each block is kept as a skip connection.

### Middle Block

The middle block is a residual block, a transformer and another residual block, run on the tensor at its smallest size.
It does not change the shape of the tensor.

### Output Blocks

The output blocks mirror the input blocks, using upsampling convolutions to double the height and width instead of halving it.
Before each block, a skip connection is joined onto the tensor as extra channels, taken in reverse order so that each output block gets the input block at the same size.

### Out

A final normalisation, activation and convolution reduce the channels back down to those of the latent.

## Generation

Once trained, you can use a UNet to remove noise from an image, or generate an entirely new image.

The UNet's output is a tensor the same shape as the latent, holding its prediction of the unit noise that was mixed in - before that noise was scaled down by the noise level.
Since the latent is the image and the noise mixed in known proportions, the predicted image follows directly: remove the scaled predicted noise and scale what is left back up.

If you start with pure noise, it will do its best to identify a plausible image that could correspond to the given conditioning.
