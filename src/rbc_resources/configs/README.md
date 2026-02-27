# Configs

Configuration files used by external neuroimaging tools.

## FSL FLIRT BBR schedule

`flirt_bbr_schedule.sch` is the FSL FLIRT boundary-based registration (BBR)
schedule file. It defines the optimization stages used when `--cost bbr` is
passed to FLIRT for BOLD-to-T1w coregistration.

Originally shipped with FSL 6.0 as `$FSLDIR/etc/flirtsch/bbr.sch`. Bundled
here so the pipeline works without a local FSL installation.
