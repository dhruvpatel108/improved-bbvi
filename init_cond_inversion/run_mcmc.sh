#!/bin/bash

# Define the parameter values
seed_values=("1" "2" "3" "4" "5")
prop_var_values=("0.01" "0.1" "1")

# Loop through each combination of parameter values
for ss in "${seed_values[@]}"
do
  for nn in "${prop_var_values[@]}"
  do
    # Print the parameter values
    echo "seed_no: $ss"
    echo "mcmc_prop_var: $nn"
    
    # Update the parameter values in the YAML file
    sed -i "s/seed_no:.*/seed_no: $ss/" params.yml
    sed -i "s/mcmc_prop_var:.*/mcmc_prop_var: $nn/" params.yml

    # Run the Python script with the modified YAML file
    python mcmc_sampler.py --config params.yml
  done
done
