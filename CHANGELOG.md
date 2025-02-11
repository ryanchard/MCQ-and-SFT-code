# Changelog

## v1.2 11-Feb-2025 (CeC)
- added jq to environment for easy reading json
- added step to check what models are runnign prior to firing off generate\_mcqs
- verified (on MacOS CLI) that the entire workflow example, steps 1-8, works
  though there are various errors to be expected (with imperfect data).

## v1.1 10-Feb-2025 (CeC)
- alcf\_inference\_utilities.py - Modified to exit with simple error message in cases where
  no network path available (such as not being local to ALCF or on VPN), avoiding
  100 lines of traceback.
- model\_access.py - Added code to centralize several key shared across multiple .py scripts,
  including ALCF\_ACCESS\_TOKEN and ALCF\_CHAT\_MODELS.
- environment.yml - Added to enable creation of conda env with all dependencies
  (many were missing and none were documented)
- generate\_mcqs\_config.yaml - Added in contemplation of easier modification of things like
  prompts, etc. but for now on hold for higher priorities.
- requirements.txt - Added to make setup easier, but deprecated in favor of using environment.yml
  which is a more comprehensive snapshot to recreate the conda environment.
