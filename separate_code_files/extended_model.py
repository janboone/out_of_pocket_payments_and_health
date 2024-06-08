#########################################################
# This file is tangled from the index.org file in the root directory
# the author runs the code from the index.org file directly in emacs
# if you do not have emacs, you can run the code to generate the trace files
# from this file
# the file expects the following folder structure to run without problems:
# the folder with the data should be located at: ./data/data_deaths_by_age_nuts_2.csv
# and the trace files are written to ./traces
#########################################################
import xarray as xr
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor
import pandas as pd
import graphviz as gr
import arviz as az
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')
from country_codes import eurostat_dictionary
from tabulate import tabulate
age_min = 35
age_max = 85
age_range = np.arange(age_max-age_min+1)[:,np.newaxis]
plot_age = np.arange(age_min,age_max+1)
first_year = 2009
last_year = 2019

df = pd.read_csv('./data/data_deaths_by_age_nuts_2.csv')

df.rename(columns={'at risk of poverty':'poverty',\
                   'percentage_material_deprivation':'deprivation',\
                   'UNMET':'unmet'},inplace=True)

df.dropna(subset=['deaths','population', 'TOOEXP',\
                  'HF3_PC_CHE','lagged_mortality'],
                    axis=0, how ='any',inplace=True)
df = df[(df.population > df.deaths) & (df.age >= age_min) & \
        (df.age <= age_max) & (df.year <= last_year) &\
        (df.year >= first_year)]
df['mortality'] = df.deaths/df.population*100 \
    # mortality as a percentage

# lagged mortality as fraction of mean lagged mortality
# per age/gender group
df['lagged_mortality_s'] = (df['lagged_mortality'])/\
    df.groupby(['age','sex'])['lagged_mortality'].\
    transform('mean')

#len(df)

df.dropna(subset=['bad_self_perceived_health'],
                    axis=0, how ='any',inplace=True)
country_index, country_list = pd.factorize(df.country,sort=True)
country_code_index, country_code_list = \
  pd.factorize(df.country_code, sort=True)
nuts2_index, nuts2_list = pd.factorize(df.nuts2,sort=True)
nuts1_index, nuts1_list = pd.factorize(df.nuts1,sort=True)
gender, gender_list =\
  np.array(pd.factorize(df.sex,sort=True),dtype=object)
year, year_list =\
  np.array(pd.factorize(df.year,sort=True),dtype=object)
age_index, age_list = \
  np.array(pd.factorize(df.age,sort=True),dtype=object)

N_countries = len(set(country_index))
N_nuts1 = len(set(nuts1_index))
N_nuts2 = len(set(nuts2_index))
N_age = len(set(age_index))

def standardize_s(x):
    x_ma = np.ma.masked_invalid(x)
    return x_ma/x_ma.std()

def standardize(x):
    return x/x.std()
# dependent variable
mortality = df.deaths.values
population = df.population.values
lagged_log_mortality = np.clip(\
    np.ma.masked_invalid(np.log(df['lagged_mortality_s'])),\
                         np.log(0.0001),np.log(10))

# nuts 2 measures
poverty_s  = (df['poverty'].values/100.0)
deprivation_s = (df['deprivation'].values/100.0)

oop_s = np.ma.masked_invalid(df['HF3_PC_CHE'].values)/100.0 # only oop
oop_e = np.ma.masked_invalid(df['HF3_PC_CHE'].values+df['HF2_PC_CHE'].values)/100.0
      # oop and voluntary insurance

too_exp = (df['TOOEXP'].values)/100.0
too_exp_lo = np.clip(np.log(too_exp/(1-too_exp)),np.log(0.0001),np.log(10))
unmet = (df['unmet'].values)/100.0



# country measures
expenditure_s = standardize(df['health expenditure per capita'].values)
std_expenditure = np.std(df['health expenditure per capita'])

# female = (df.sex == 'F').astype('uint8').values

N = len(mortality) # total sample size
N_years = len(year_list)


quality = df['infant mortality']/100.0
bad_health = df['bad_self_perceived_health']/100.0
lagged_log_mortality = np.asarray(lagged_log_mortality)
unmet = np.asarray(unmet)

coords = {"country":country_list, "nuts2":nuts2_list,\
          "gender":gender_list, "age":age_list,\
          "year":year_list}

def standardize(x):
    return x/x.std()


poverty = deprivation_s.values
oop = oop_s.values

with pm.Model(coords=coords) as extended_model:
    
    sd_fixed_effects = 0.3
    # hierarchical priors
    sd_prior_b = pm.HalfNormal('sd_prior_b', sigma = 0.1)
    σ = pm.HalfNormal('σ', sigma = 1.0)
    
    # Too Expensive equation
    ## NUTS 2 regional fixed effect:
    mu_2_too      = pm.Normal('mu_2_too', mu = -5.0,\
                              sigma = sd_fixed_effects, dims="nuts2")
    ## time fixed effect:
    mu_t_too = pm.Normal('mu_t', mu = 0.0,\
                     sigma = sd_fixed_effects, dims="year")
    ## coefficients of the TooExp equation:
    b_oop         = pm.HalfNormal('b_oop', sigma = sd_prior_b,\
                                  dims="country")
    b_interaction = pm.HalfNormal('b_interaction',\
                                  sigma = sd_prior_b, dims="country")
    mu_too_exp_lo = pm.Deterministic('mu_too_exp_lo', \
                    mu_2_too[nuts2_index] + mu_t_too[year] +\
                    expenditure_s * oop *\
                    (b_oop[country_index] +\
                     b_interaction[country_index] * poverty))
    b_health      = pm.HalfNormal('b_health', sigma = sd_prior_b)
    Too_exp_lo    = pm.Normal('Too_exp_lo', mu = mu_too_exp_lo + b_health * bad_health,\
                                sigma = σ, observed = too_exp_lo)

    
    sd_fixed_effects = 0.3
    # hierarchical priors
    sd_prior_beta = pm.HalfNormal('sd_prior_beta', sigma = 0.1)
    
    # Mortality equation
    ## age/gender fixed effect:
    beta_age = pm.Normal('beta_age', mu = -3.0,\
                         sigma = sd_fixed_effects,\
                         dims=("age","gender"),\
                         initval=-3*np.ones((N_age,2)))
    h = pm.Deterministic('h',pt.sigmoid(\
                              beta_age[age_index,gender]))
    
    ## multiplier effect: x
    ### NUTS 2 fixed effect:
    mu_2_m   = pm.Normal('mu_2_m', mu = 0.0,\
                         sigma = sd_fixed_effects, dims="nuts2")
    ### coefficients of the mortality equation:
    beta_lagged_log_mortality = pm.Normal('beta_lagged_log_mortality',\
                                          mu = 0, sigma = sd_prior_beta)
    beta_unmet = pm.HalfNormal('beta_unmet', sigma = sd_prior_beta)
    beta_poverty = pm.HalfNormal('beta_poverty', sigma = sd_prior_beta)
    mu_x = mu_2_m[nuts2_index] + beta_unmet*unmet +\
                                 beta_poverty*poverty+\
                                 beta_lagged_log_mortality*\
                                 lagged_log_mortality
    beta_quality = pm.Normal('beta_quality', mu = 0,\
                                  sigma = sd_prior_beta)
    beta_health = pm.Normal('beta_health', mu = 0,\
                                  sigma = sd_prior_beta)
    x = pm.Deterministic('x', mu_x+beta_health*bad_health+\
  				   beta_quality*quality)
    ##  combining h and x
    flat_exp = pt.switch(
      pt.lt(x, 0.7), # if
      pt.exp(x), # then
      pt.exp(0.7*(x/0.7)**0.1) # else
      )
    mortality_function = h*flat_exp
    
    ## equation binomial distribution number of deaths:
    m = pm.Deterministic('m', mortality_function)
    obs = pm.Binomial("obs", p = m,\
                      observed=mortality, n = population)


with extended_model:
    idata_extended = pm.sample(target_accept=0.85)
    # pm.sampling_jax.sample_numpyro_nuts(idata_baseline, extend_inferencedata=True)
    pm.sample_posterior_predictive(idata_extended, extend_inferencedata=True)

idata_extended.to_netcdf("./traces/extended_model.nc")
