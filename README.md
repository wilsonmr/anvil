# Git repo for flow-based generative model project

We can change this description and title of project as we go along

 - `norm_flow_pytorch.py`

    contains the model class and the loss function in the `pytorch` framework
 - `train_example.py`

    a simple script which demonstrates how to use the model to learn a
    toy distribution

 - `phi_four.py`

   trains a model for the phi^4 theory and generates a sample of the required number of 
   configurations.

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

![ratio plot of sampled vs. target covariance](./example_output/ratio.png)

## phi^4 example

The script can be run either by training a new model, or by loading a model. To train, run
```bash
$ python3 phi_four.py train <new_model_name>.pt
```
or to load an exisitng model, run
```bash
$ python3 phi_four.py load <model_name>.pt
```

The models will be saved to/loaded from a `models` directory. The output will be a set of field configurations - each row of the output matrix corresponds to a flattened configuration. The number of configurations generated is set by `target_length` in the `main` method.

### Some notes...
* I think there's currently some unecessary code in the `array_mapping` class - I need to tidy this up.
* I have only trained a 4x4 model so far (around 5 minutes) - I started a 6x6 model but this would take around 2 hours 
so I will do this soon.
* Based on a few runs of the code, time to run the MH sampling seems to increase linearly with target number of configurations up until ~10^4, after which it seems to diverge.
* The fraction of update proposals accepted seems to stay in the range of 0.3-0.5 regardless of target number of configurations, but again note this is only based on a few runs of the code for now.
* I think a *lot* of fine-tuning will need to be performed!