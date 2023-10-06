**Score: 6.5 / 10**

* Good summary of code-review updates.

## Jupyter notebook demonstration
* Missing. You were supposed to include a cleaned-up notebook demonstrating your code's functionality.

## single_neuron_training.jl
* Why did you rename `predict` to `predict2`? That breaks your training!
* Your training is also broken because you override the `gradient` function with a variable called `gradient`.
* lines 88-90: these are redundant with the ones that actuall check for an input-vector.
* line 55: this needs to divide by number of data ponts!
* lines 47, 49: your loops are backwards.
  - This is inefficient since you call `predict` on the same point many times.
  - It also gives the wrong value for the bias.
