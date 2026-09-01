# VAE

A VAE is a method for compressing a high-dimensional object (an image in this case) to a lower dimensional representation, and vice versa.

## Autoencoders

An autoencoder is a pair of neural networks which take some high dimensional object and map them to a lower dimensional representation in 'latent space'.
One network maps the object to latent space, the other network maps the latent representation back to the higher-dimensional representation.
If the network is trained correctly, the result should be close to the original in every way that matters.

You can also sample from the latent space - taking any point in that space and generating a new object in the original space.

Because the latent space has fewer dimensions, it is less computationally intensive to do calculations on the object in this representation.

## Variational Autoencoders

The main problem with standard autoencoders is that they don't make full use of the available latent space.
'Real' inputs can be clustered together and often overlap, while a lot of the latent space doesn't map back to anything that makes sense.

A variational autoencoder solves this by applying two loss terms when training - how closely the two networks match (as with standard autoencoders) and how normally distributed the training examples are in latent space.
Rather than outputting a specific point in the latent space, it outputs a probabilistic region (in the form of a mean and a variance around that mean).
This forces the networks to spread the points out.
When using the encoder for inference, only the mean is typically used.

## Image VAEs

In latent diffusion, VAEs are used to create a smaller representation of images that is easier to work with - smaller spatial dimensions (though with more channels per location).

There is no fixed architecture for VAEs in general - you can structure the network any way you like.
But for image0representing VAEs there is a standard architecture for both the encoder and the decoder, and they are broadly symmetrical.

### Encoding

The encoder network:

1. Runs the image tensor through an initial convolution layer.
2. Runs the tensor through a series of 'down' levels, which halve the size of the image at each block using ResNets.
3. Runs the diminished tensor through a series of 'mid' layers, which use attention and ResNet without changing the shape of the tensor.
4. Performs a final normalisation and convolution to get a mean position and distribution for each point.
5. Discards the distributions to produce a final latent tensor.

### Decoding

The decoder network:

1. Runs the latent tensor through an initial convolution layer.
2. Runs the tensor through a series of 'mid' layers, which use attention and ResNet without changing the shape of the tensor.
3. Runs the refined tensor through a series of 'up' levels, which double the size of the image at each block using ResNets.
4. Performs a final normalisation and convolution to produce a final image tensor.
