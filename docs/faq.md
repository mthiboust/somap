# FAQ

### Why the input data is passed as a dictionary with the "bu_v" key?

I intend to add support for other kind of data in the future. A more complex variant of SOM may take additional inputs that could be categorized into:

* ***"bu"***: Bottom-up driving input (the classic input of SOMs)
* ***"lat"***: Lateral contextual input which could influence the processing before selecting the BMU.
* ***"td"***: Top-down modulatory input which could influence the selection of the BMU.

The ***"v"*** and ***"m"*** suffixes correspond respectively to ***value*** and ***mask*** for supporting weighted inputs.


### I want to keep track of the auxilary data of my model across different calls to `smp.make_step` or `smp.make_steps`

You can keep track of the auxilary data of your models in a list that is merged at the end:

```python
epoch = 10
auxs = []
for i in range(0, epoch):
    model, aux = smp.make_steps(model, {"bu_v": data})
    auxs.append(aux)

# Concatenate the 'aux' outputs if there are several
aux = jax.tree_util.tree_map(
    lambda x, *y: np.concatenate((x, *y), axis=0),
    *auxs,
    is_leaf=lambda x: isinstance(x, list),
)
```

