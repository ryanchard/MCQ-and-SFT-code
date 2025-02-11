# Changelog


## 10Feb (CeC)
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
