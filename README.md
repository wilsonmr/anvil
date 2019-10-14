# Git repo for flow-based generative model project

We can change this description and title of project as we go along

 - `norm_flow_pytorch.py`

    contains the model class and the loss function in the `pytorch` framework
 - `train_example.py`

    a simple script which demonstrates how to use the model to learn a
    toy distribution

## Toy example

In order to run the toy example you will need a python environment with `python`
3.7, `pytorch`, `tqdm` and some of the other common 3rd party modules like
`numpy` and `matplotlib`.

Once you have this, simple run

```bash
$ ./train_example.py
```

which should train the model, and output the plots into `./example_output/`.
If the model successfully trains, then the ratio of the covariance matrix
sampled from the trained model over the input covariance should be approximately
1

[ratio plot of sampled vs. target covariance](./example_output/ratio.png)
