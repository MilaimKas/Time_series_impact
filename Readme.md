# Time serie impact 

## What is this ?

Set of modules to perform in depth analysis of time series and intervention simulation to assess the power of a model.

## How to install

```
pip install --extra-index-url=https://rspm.parship.internal/python-repo/latest/simple TimeSeries_impact
```

## Example usage

Import modules
```python 

from TimeSeries_impact import utilities, ts_analysis, impact, plot_functions

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

Example, fake, dataset
```python

ts = utilities.make_time_serie(200, freq=[7], nbr_rand_event=5)
data = pd.DataFrame()
data["target"] = ts["obs"]
for i in range(len(ts["control"])):
    data[f"control {i}"] = ts["control"][i]

data.plot()
plt.show()
```

Analyse time series and components
```python
# create TSA object
reload(ts_analysis)
TSA = ts_analysis.TSA(data)

# check normalized and shifted ts
TSA.plot_scaled_view()

# plot component from decomposition
TSA.plot()
TSA.plot_component()

# analyse the similarity of the components
print()
print("Analysis results")
res_res, res_seas = TSA.analyze()
corr = TSA.correlation()

print()
print("Similarity of residuals")
print(res_res)

print()
print("Similarity of seasonality")
print(res_seas)

print()
print("Correlation of the time series")
print(corr)
```

Perform simulations to asses the power of the model
```python

# create impact class
impact_class = impact.SimImpact(data)

# single simulation
relup = np.linspace(0.01, 0.2, 10)
impact_class.make_sim(relup_list=relup, test_size=14)
fig = impact_class.plot_sim_rel()
display(fig)

# check single causalimpact results with relup as keys
print("Keys", impact_class.res_sim.keys())
impact_class.res_sim['0.0944'][1].plot()

# power analysis by looping over different pre- and post- period lengths
_ = impact_class.power_analyse(relup_list=relup)
fig_power = impact_class.plot_power(alpha=[5,10,20])
display(fig_power)
```


Perform classic causal impact analysis with the right python package (tfcausalimpact)
```python
# perform classic causalimpact with tfcausalimpact

# add effect
relup = 0.2
uplift = relup*np.mean(data["target"])
data["target"] = utilities.add_effect(data["target"], uplift, 50)

pre_period = [0, 149]
post_period = [150, 199]

ci = impact.CausalImpact(data, pre_period, post_period, model_args={"period":7})
print(ci.summary())
print(ci.summary(output='report'))
ci.plot()
```