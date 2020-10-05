import os




def is_herring():
    if "RUN_HERRING" in os.environ and os.environ["RUN_HERRING"] == '1':
        return True
    else:
        return False
