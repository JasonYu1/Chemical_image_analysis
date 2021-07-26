import tkinter as tk
from tkinter import ttk, Canvas
import os
from tkinter.filedialog import askopenfilename
import matlab.engine
import tifffile
from PIL import Image, ImageTk
import numpy as np
import threading
import time
from tqdm import tqdm
#from PIL.Image import core as _imaging
import progressbar as pb


# set gui params


class Main(tk.Tk):
    """Simply GUI Structure"""

    def __init__(self):
        """Initialization"""
        tk.Tk.__init__(self)
        self.geometry('100x100')
        self.resizable(0, 0)
        # self.minsize(120, 1)
        # self.maxsize(1200, 845)
        self._frame = None
        self.switch_frame(file_page)

    def switch_frame(self, frame_class):
        """switch frame button"""
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()


class file_page(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, height=800, width=800)  # need to define height and width to use .place()
        self.master.geometry('300x120')
        self.master.title('Chemical Image Analysis')
        x_position = 80
        tk.Label(self, text='PSNR: ').place(x=x_position, y=20)
        tk.Label(self, text='SSIM: ').place(x=x_position, y=50)
        tk.Label(self, text='RMSE: ').place(x=x_position, y=80)



if __name__ == "__main__":
    app = Main()
    #app.title("Denoising v0.1")
    app.mainloop()