# CLIP

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