from ctypes import windll
user32 = windll.user32
user32.SetProcessDPIAware()

import tkinter
root = tkinter.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()
print(width,height)