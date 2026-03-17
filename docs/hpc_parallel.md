# Parallel Execution on HPC

RBC processes subjects sequentially by default. For large datasets, you can
parallelize across subjects using your cluster's job scheduler. Each subject
is fully independent, so this is embarrassingly parallel.

## Slurm Job Arrays

Job arrays are the simplest way to run one subject per job. Create a text file
listing participant labels, then index into it with `$SLURM_ARRAY_TASK_ID`.

### 1. Create a participant list

```bash
# List all participant labels (without the sub- prefix)
ls -d /data/bids/sub-* | xargs -n1 basename | sed 's/sub-//' > participants.txt
```

### 2. Write a Slurm batch script

```bash
#!/bin/bash
#SBATCH --job-name=rbc
#SBATCH --array=1-$(wc -l < participants.txt)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rbc_%A_%a.out

PARTICIPANT=$(sed -n "${SLURM_ARRAY_TASK_ID}p" participants.txt)

rbc /data/bids /data/output all \
    --runner docker \
    --participant-label "$PARTICIPANT"
```

Save this as `rbc_array.sbatch` and submit:

```bash
mkdir -p logs
sbatch rbc_array.sbatch
```

### 3. Monitor progress

```bash
# Check running/pending jobs
squeue -u $USER

# Check a specific job's output
cat logs/rbc_<jobid>_<arrayid>.out
```

## GNU Parallel (single node)

If you have a multi-core machine without a scheduler:

```bash
cat participants.txt | parallel -j 4 \
    rbc /data/bids /data/output all \
        --runner docker \
        --participant-label {}
```

`-j 4` runs 4 subjects concurrently. Adjust based on available memory (expect
roughly 8-16 GB per subject).

## PBS/Torque Job Arrays

```bash
#!/bin/bash
#PBS -N rbc
#PBS -J 1-100
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=04:00:00

PARTICIPANT=$(sed -n "${PBS_ARRAY_INDEX}p" participants.txt)

rbc /data/bids /data/output all \
    --runner docker \
    --participant-label "$PARTICIPANT"
```

## Using local scratch for intermediates

RBC writes a lot of intermediate files during processing. On HPC clusters,
the shared filesystem (Lustre, GPFS, NFS) can become a bottleneck. Most
clusters provide fast local scratch storage on each compute node. Use
`--tmp-dir` to point intermediates there:

```bash
rbc /data/bids /data/output all \
    --runner singularity \
    --tmp-dir /lscratch/$SLURM_JOB_ID \
    --participant-label "$PARTICIPANT"
```

Common scratch paths by cluster environment:

| Environment | Typical scratch path |
|---|---|
| Slurm (NIH Biowulf-style) | `/lscratch/$SLURM_JOB_ID` |
| Slurm (generic) | `$TMPDIR` or `/tmp` |
| PBS/Torque | `$TMPDIR` |
| SGE | `$TMPDIR` |

If your cluster doesn't allocate local scratch automatically, you may need to
request it (e.g., `#SBATCH --gres=lscratch:100` for 100 GB on Biowulf).

Scratch is cleaned up when the job ends, so only final outputs (written to
`output_dir`) are preserved.

## Tips

- **One subject per job** is the recommended granularity. RBC's internal
  processing is sequential per subject, so splitting finer than that won't help.
- **Memory:** 16 GB per subject is a rough starting point. Increase to 32 GB for
  data with high spatial resolution or long functional runs. These numbers may
  be revised after memory profiling (#93).
- **Walltime:** Expect 1-3 hours per subject depending on the number of
  functional runs and hardware.
- **Singularity on HPC:** Most clusters don't allow Docker. Use
  `--runner singularity` instead.
- **Shared filesystem:** Make sure `input_dir` and `output_dir` are on a
  filesystem accessible to all compute nodes (e.g., Lustre, GPFS, NFS).
- **Local scratch:** Use `--tmp-dir` to place intermediate files on fast
  node-local storage. This can significantly reduce I/O wait times.
- **Failed jobs:** Re-run only failed subjects by filtering the participant list:
  ```bash
  # Find subjects without output and re-run
  comm -23 <(sort participants.txt) \
           <(ls /data/output/sub-* -d | xargs -n1 basename | sed 's/sub-//' | sort) \
      > participants_failed.txt
  ```
