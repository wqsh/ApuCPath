NAME_SOFTWARE = "ApuCell"

NAME_SOFTWARE_LOG = f"[{NAME_SOFTWARE}] "

NAME_CONFIG = f"{NAME_SOFTWARE}_config.txt"

# ==============这块是注册的参数==============

NAME_FOR_MSREGISTER = ("MSRegisterExportDLL.dll",
                       f"{NAME_SOFTWARE}.license",
                       f"{NAME_SOFTWARE}.probation")

LINK_APPLICATION = "https://www.hao123.com"

NAME_PUBLIC_SECURITY = "MSSource.dll"

# ==============这块是注册的参数==============

# ==============这块是日志的参数==============

FOLDER_CONFIG = "config"

FOLDER_MSREGISTER = "MSRegister"

FOLDER_LOG = "LOG"

PATH_LOG = f'{NAME_SOFTWARE}.log'
PATH_ERROR = f'{NAME_SOFTWARE}.error'
PATH_STATE = f'{NAME_SOFTWARE}.state'

EXPIRATION_TIME = {'Year': 2026, 'Month': 7, 'Day': 31}

INFO_TO_USER_Staff = (
    f'\n[{NAME_SOFTWARE}] {NAME_SOFTWARE} is expired! Please send e-mail to  for the new version.',
    f'\n[{NAME_SOFTWARE}] Warning! The current license will expired in 7 days. Please send e-mail to for the new version.',
    f'\n[{NAME_SOFTWARE}] Starting...',
    f'\n[{NAME_SOFTWARE}] Finished!',
    f'\n[{NAME_SOFTWARE}] Writing config file...',)

# ==============这块是日志的参数==============

class CSetSystemGlobal:

    @staticmethod
    def SetNameSoftware(name_software:str):

        global NAME_SOFTWARE
        global NAME_SOFTWARE_LOG
        global NAME_CONFIG
        global NAME_FOR_MSREGISTER
        global PATH_LOG
        global PATH_ERROR
        global INFO_TO_USER_Staff

        NAME_SOFTWARE = name_software

        NAME_SOFTWARE_LOG = f"[{NAME_SOFTWARE}] "

        NAME_CONFIG = f"{NAME_SOFTWARE}_config.txt"

        NAME_FOR_MSREGISTER = ("MSRegisterExportDLL.dll",
                               f"{NAME_SOFTWARE}.license",
                               f"{NAME_SOFTWARE}.probation")

        PATH_LOG = f'{NAME_SOFTWARE}.log'
        PATH_ERROR = f'{NAME_SOFTWARE}.error'

        INFO_TO_USER_Staff = (
            f'\n[{NAME_SOFTWARE}] {NAME_SOFTWARE} is expired! Please send e-mail to  for the new version.',
            f'\n[{NAME_SOFTWARE}] Warning! The current license will expired in 7 days. Please send e-mail to for the new version.',
            f'\n[{NAME_SOFTWARE}] Starting...',
            f'\n[{NAME_SOFTWARE}] Finished!',
            f'\n[{NAME_SOFTWARE}] Writing config file...',)