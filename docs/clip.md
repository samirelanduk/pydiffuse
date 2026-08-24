# CLIP

CLIP is a method for measuring how closely an image matches a particular text description.

It does this by creating a shared vector space that both images and text can be mapped to - if the vector for an image has a very small angle with the vector for a description, the description matches the image closely.

During training, the language/image pairs are run through a neural network, whose weights are then optimised to produce similar vectors for the language/image pairs in the training set, and dissimilar vectors for mismatched pairs (hence contrastive). During inference images and text can be run through this same trained network to get the representative vectors.

For image diffusion models, we only make use of the text encoding - turning a piece of text into a vectors that represent the semantic meaning of each token.
We do this by (1) turning the text into a list of integer tokens via a fixed mapping, (2) looking up the vector representation of each individual token, and then (3) creating a final encoding by using multi-headed attention to adjust each vector based on all the tokens before it.

## Tokenising

Tokenising is a process which converts any string to a list of integers that represent that string.
Each integer represents a substring within the original string (which could be a whole word, or a part of a word, or even a single character).
There are certain special tokens which represent the start/end of the string.

The mapping of substrings to token integers comes from a prebuilt 'tokeniser' directory.
The library comes with one, but you can provide your own.

The final result is a list of integers which represent the original string, as a list of lists where each list is of length 77.

## Embedding

Embedding takes the tokens (a list of list of token integers) and maps each token to a vector, to produce a list of list of vectors.

Each vector is a pre-trained representation of that token in isolation, with a small adjustment for its position within the list.
It contains no adjustment for meaning context within the prompt.

The vectors come from a tensor of vectors you must supply.
Each token has its own pre-trained vector, and each of the 77 positions has its own small 'adjustment' vector that is added to the token vector.
The final result is one embedding vector per token that represents that token at that position, in isolation.

## Encoding

Encoding produces one vector per token, but where each vector represents that token in the context of all the tokens before it.

This is done via a series of layers, where in each layer a multi-head attention process is applied, then a standard MLP layer, using supplied weights and biases.
At the end of this process, the vectors have been updated with information from the previous tokens.

A final normalisation layer is then applied.