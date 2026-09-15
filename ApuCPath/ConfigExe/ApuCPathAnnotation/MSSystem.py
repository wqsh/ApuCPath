import sys
if sys.platform.startswith('win'):
    NAME_EXE = r"ApuCPathAnnotation.exe"
elif sys.platform.startswith('linux'):
    NAME_EXE = r"ApuCPathAnnotation"
else:
    NAME_EXE = ""