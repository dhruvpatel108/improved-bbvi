#!/bin/bash

# Define the parameter values
bs_values=("25")
nf_values=("150" "1500" "15000" "150000" "375000" "750000" "1500000")
#niter_values=("0.01" "0.1" "1")

# Loop through each combination of parameter values
for bs in "${bs_values[@]}"
do
  for nn in "${nf_values[@]}"
  do
    # Print the parameter values
    echo "batch_size: $bs"
    echo "n_forward: $nn"
    let "n_iter = $nn / $bs"
    echo "n_iter: $n_iter"
    
    # Update the parameter values in the YAML file
    sed -i "s/batch_size:.*/batch_size: $bs/" params.yml
    sed -i "s/n_iter:.*/n_iter: $n_iter/" params.yml
    sed -i "s/elbo_batch_size:.*/elbo_batch_size: 2000/" params.yml

    # Run the Python script with the modified YAML file
    python bbvi.py --config params.yml
  done
done
