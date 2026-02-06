# C-PAC Test Comparison Run

Run C-PAC on the test dataset with minimal parallelization:

```sh
docker run --rm -it \
  -v $(pwd)/tests/data/ds000001:/bids_input:ro \
  -v $(pwd)/tests/data/cpac_outputs/ds000001:/outputs \
  fcpindi/c-pac:release-v1.8.5.dev1 \
  /bids_input /outputs participant \
  --n_cpus 1 \
  --num_ants_threads 1 \
  --skip_bids_validator \
  --preconfig rbc-options \
  --save_working_dir
```

This mounts the BIDS dataset as read-only input and saves outputs and intermediates locally.

**Note:** For proper comparison testing, use the same container version when setting up NiWrap.

Clean up outputs:

```sh
rm -rf tests/data/cpac_outputs/ds000001/
```
