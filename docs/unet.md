# UNets

A UNet is a neural network architecture, used here to predict denoised images from noised images.

In latent diffusion, you provide a UNet with (1) a latent image tensor with some amount of noise applied, and (2) a number indicating how much noise has been applied, and train it to predict the original image.
Usually a third input is provided too, a CLIP vector representing a text description of the image (the conditioning).

You provide images at various levels of noising so that the model can deal with very noisy images and barely noisy images.
To do this you have to decide how many noise levels you want, and what noise fractions each should have.

The original latent diffusion model had 1000 noise levels - 0 to 999, with this number referred to as t (timestep).
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