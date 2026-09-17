# Sampling

A UNet can make a prediction of the image underneath a layer of noise, and can even go from noise to an image in a single jump - though usually this will result in a very poor image.

Generally UNets are used to generate images by being run repeatedly as part of an iteration towards the final image.

## Sampling Loop

You start with a noise schedule, which is the noise levels you will denoise to at each stage.
You also need a latent image full of random noise, that will be progressively denoised.

For each noise level, you run the UNet by giving it the noised latent, the current noise level, and a conditioning.
You run it on the conditioning of what the image should contain (the positive conditioning) and a conditioning that describes either images in general, or a specific negative description.

You get back two noise predictions - one pointing towards the image you want, and one pointing towards the 'real' images in general.
In order to emphasise the image you want rather than images in general, you start from the negative prediction and move along the difference between the two towards the positive prediction, by some multiplier called CFG (Classifier Free Guidance).
A CFG of 1 gives you just the positive prediction, and higher values go past it, further away from the negative one.
This gives you a noise prediction indicative of the specifics of your prompt.

What you actually do with this prediction can vary depending on which 'sampling' algorith you use.

## Algorithms

There are various ways of creating a new latent image from a UNet prediction at each loop iteration.

### Euler

The Euler algorithm is one of the simpler sampling algorithms.
You take the prediction of the denoised image, and add the predicted noise back, but scaled to the next noise level.
So if you are on noise level 0.8, and the next is 0.7, you run the UNet twice at 0.8, get the predicted noise, remove the noise from the latent to get a prediction from 0.8, then re-add the noise to get a latent scaled to 0.7.

### Heun

The Heun algorithm is similar to Euler, and begins the same way.
However once you have your predicted image for that noise level, it then runs the UNet a second pair of times at the new noise level to get a new predicted noise, and uses the average of the two noise predictions to create a final latent image for that iteration.