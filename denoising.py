import tkinter as tk
from tkinter import ttk, Canvas
import os
from tkinter.filedialog import askopenfilename, askdirectory
import matlab.engine
import tifffile
from PIL import Image, ImageTk
import cv2
import shutil
import numpy as np
from distutils.dir_util import copy_tree



# set gui params
font = 'Arial'
gui_size = '520x400'


class Main(tk.Tk):
    """Simply GUI Structure"""

    def __init__(self):
        """Initialization"""
        tk.Tk.__init__(self)
        self.geometry(gui_size)
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

    def user_manual(self):
        os.system('User_Manual.pdf')

    def export(self):
        directory = askdirectory()
        if denoising is True:
            if denoising_method == 'Bm4d':
                folder = '/denoise_bm4d/'
            if denoising_method == 'STV':
                folder = '/denoise_stv/'
            copy_tree("." + folder, directory +'/' + folder)
        if decomposition is True:
            if decomposition_method == 'MCR':
                folder = '/mcr_chemical_maps/'
            if decomposition_method == 'LS':
                folder = '/ls_chemical_maps/'
            copy_tree("." + folder, directory +'/'+ folder)


class file_page(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, height=800, width=800)  # need to define height and width to use .place()
        self.master.geometry('520x400')
        self.master.title('Chemical Image Analysis')
        img_x = 280
        img_y = 95
        try:
            os.path.exists('uint8.tif')
        except ValueError:
            pass

        shutil.rmtree('./denoise_bm4d', ignore_errors=True)
        shutil.rmtree('./denoise_stv', ignore_errors=True)
        shutil.rmtree('./ls_chemical_maps', ignore_errors=True)
        shutil.rmtree('./mcr_chemical_maps', ignore_errors=True)

        def open_file():
            """Open a file for editing"""

            if ent_lbl.get() == '':
                global filepath
                filepath = askopenfilename(
                    filetypes=[("All Files", "*.*"), ("Tif File", "*.tif"), ("Txt File", "*.txt")]
                )
                if not filepath:
                    return
                ent_lbl_text.set(filepath)
            else:
                filepath = ent_lbl.get()

            # global variables used later
            global directory, filetype, frame_number, filename
            directory = os.path.dirname(os.path.abspath(filepath))
            filename_w_ext = os.path.basename(filepath)
            filename = os.path.splitext(filename_w_ext)[0]
            filetype = os.path.splitext(filename_w_ext)[1]


            eng = matlab.engine.start_matlab()

            if filetype == '.tif':
                print('The tif file is '+ str(tifffile.imread(filepath).dtype))
                if tifffile.imread(filepath).dtype == 'uint16' or 'uint32' or 'float32':
                    img = tifffile.imread(filepath)
                    if len(img.shape) == 2:
                        dst = np.zeros(shape=(img.shape[0], img.shape[1]))
                    elif len(img.shape) == 3:
                        dst = np.zeros(shape=(img.shape[1], img.shape[2]))
                    raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
                    im = np.uint8(raw)
                    imlist = []
                    for m in im:
                        imlist.append(Image.fromarray(m))
                    imlist[0].save("uint8.tif", save_all=True, append_images=imlist[1:])

            global K, M, N
            if filetype == '.tif':
                A = eng.size(eng.imread(filepath))
                M = int(A[0][0])
                N = int(A[0][1])
                K = int(eng.get_frame(filepath))
                instruction["text"] = 'The dimensions of the file is ' + str(M) + ' x ' + str(N) + ' x ' + str(K) + '.'

                def sel(self):
                    img = tifffile.imread(filepath)
                    if len(img.shape) == 2:
                        load_raw = Image.fromarray(data_raw)
                    elif len(img.shape) == 3:
                        load_raw = Image.fromarray(data_raw[int(var.get())])
                    width, height = load_raw.size
                    ratio = M / 200
                    # load_raw = load_raw.resize((int(width / ratio), int(height / ratio)), Image.ANTIALIAS)
                    load_raw = load_raw.resize((int(width / ratio), int(height / ratio)), Image.NEAREST)
                    render_raw = ImageTk.PhotoImage(load_raw)
                    img_raw.configure(image=render_raw)
                    # img_raw = tk.Label(self, image=render_raw)
                    img_raw.image = render_raw
                    img_raw.place(x=img_x, y=img_y)

                global var
                var = tk.DoubleVar(self)
                tk.Scale(self, variable=var, orient=tk.HORIZONTAL, from_=0, to=K - 1, length=200,
                         command=sel).place(x=img_x, y=295)
                var.set(int(K / 2))
                global data_raw
                img = tifffile.imread(filepath)
                if len(img.shape) == 2:
                    dst = np.zeros(shape=(img.shape[0], img.shape[1]))
                elif len(img.shape) == 3:
                    dst = np.zeros(shape=(img.shape[1], img.shape[2]))
                data_raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
                if len(img.shape) == 2:
                    load_raw = Image.fromarray(data_raw)
                elif len(img.shape) == 3:
                    load_raw = Image.fromarray(data_raw[int(K / 2)])

                width, height = load_raw.size
                ratio = M / 200
                # load_raw = load_raw.resize((int(width/ratio), int(height/ratio)), Image.ANTIALIAS)
                load_raw = load_raw.resize((int(width/ratio), int(height/ratio)), Image.NEAREST)
                render_raw = ImageTk.PhotoImage(load_raw)
                global img_raw
                img_raw = tk.Label(self, image=render_raw)
                img_raw.image = render_raw
                img_raw.place(x=img_x, y=img_y)
            elif filetype == '.txt':
                A = eng.size(eng.importdata(filepath))

                M = int(A[0][0])
                N = int(A[0][1])
                K = int(N/M)
                instruction["text"] = 'The dimensions of the file is ' + str(N) + ' x ' + str(M) + '. Click "Transform" to perform a 3D transformation.'
                tk.Button(self, text='Transform', command=transform).place(x=100, y=363)
            eng.quit()

        instruction = tk.Label(self, text='')
        instruction.place(x=10, y=50)

        def transform():
            eng = matlab.engine.start_matlab()
            #eng.txt_to_tif(filepath, nargout=0)
            eng.txt2tiff_32(filepath, nargout=0)
            eng.quit()

            def sel(self):
                # load_raw = Image.fromarray(data_raw[int(var.get())].astype('uint8'))
                dst = np.zeros(shape=(M, N))
                data_raw = tifffile.imread(filename + '.tif')
                data_raw = cv2.normalize(data_raw, dst, 0, 255, cv2.NORM_MINMAX)
                load_raw = Image.fromarray(data_raw[int(var.get())])
                width, height = load_raw.size
                ratio = M / 200
                load_raw = load_raw.resize((int(width / ratio), int(height / ratio)), Image.ANTIALIAS)
                render_raw = ImageTk.PhotoImage(load_raw)
                img_raw.configure(image=render_raw)
                # img_raw = tk.Label(self, image=render_raw)
                img_raw.image = render_raw
                img_raw.place(x=img_x, y=img_y)

            global var
            var = tk.DoubleVar(self)
            tk.Scale(self, variable=var, orient=tk.HORIZONTAL, from_=0, to=K - 1, length=200,
                     command=sel).place(x=img_x, y=295)
            var.set(int(K / 2))
            dst = np.zeros(shape=(M, N))
            data_raw = tifffile.imread(filename + '.tif')
            data_raw = cv2.normalize(data_raw, dst, 0, 255, cv2.NORM_MINMAX)
            load_raw = Image.fromarray(data_raw[int(K / 2)])
            width, height = load_raw.size
            ratio = M / 200
            load_raw = load_raw.resize((int(width / ratio), int(height / ratio)), Image.ANTIALIAS)
            render_raw = ImageTk.PhotoImage(load_raw)
            global img_raw
            img_raw = tk.Label(self, image=render_raw)
            img_raw.image = render_raw
            img_raw.place(x=img_x, y=img_y)

        def Next():
            global denoising, decomposition, decomposition_method, denoising_method, batch
            if n_multi.get() == 1:
                batch = 1
            elif n_multi.get() == 0:
                batch = 0
            denoising = False
            decomposition = False
            if n_denoising.get() == 1:
                denoising = True
            if n_decom.get() == 1:
                decomposition = True
            if n_denoising_bm4d.get() == 1:
                denoising_method = 'Bm4d'
            if n_denoising_stv.get() == 1:
                denoising_method = 'STV'
            if n_decom_sp.get() == 1:
                decomposition_method = 'Spectral Phasor'
            if n_decom_mcr.get() == 1:
                decomposition_method = 'MCR'
            if n_decom_ls.get() == 1:
                decomposition_method = 'LS'

            if n_denoising.get() == 0 and n_decom.get() == 1:
                if n_decom_mcr.get() == 1:
                    master.switch_frame(decomposition_mcr)
                if n_decom_sp.get() == 1:
                    master.switch_frame(decomposition_phasor)
                if n_decom_ls.get() == 1:
                    master.switch_frame(decomposition_ls)
            else:
                if n_denoising_bm4d.get() == 1:
                    master.switch_frame(denoising_bm4d)
                if n_denoising_stv.get() == 1:
                    master.switch_frame(denoising_stv)

        def multi_clicked():
            if n_multi.get() == 1:
                if ent_lbl_text.get() == '':
                    instruction["text"] = 'Choose a random file in the folder for batch processing.'
            elif n_multi.get() == 0:
                if ent_lbl_text.get() == '':
                    instruction["text"] = ''


        # filepath entry
        ent_lbl_text = tk.StringVar()
        ent_lbl_text.set('')
        ent_lbl = tk.Entry(self, width=58, textvariable=ent_lbl_text)
        ent_lbl.place(x=94, y=14)

        # multifile checkbox
        n_multi = tk.IntVar()
        n_multi_chk = ttk.Checkbutton(self, text='Batch', variable=n_multi, command=multi_clicked)
        n_multi_chk.place(x=462, y=15)
        n_multi.set(False)

        # File selection buttons

        select_btn = tk.Button(self, text="Select File", command=open_file)
        select_btn.place(x=10, y=10)

        # Separator
        separator1 = ttk.Separator(self, orient='horizontal')
        separator1.place(x=0, y=80, relwidth=1, relheight=1)

        def denoising_clicked():
            if n_denoising.get() == 1:
                n_denoising_stv_chk.state(['!disabled'])
                n_denoising_bm4d_chk.state(['!disabled'])
            elif n_denoising.get() == 0:
                n_denoising_stv_chk.state(['disabled'])
                n_denoising_bm4d_chk.state(['disabled'])

        def bm4d_clicked():
            if n_denoising_bm4d.get() == 1:
                n_denoising_stv_chk.state(['disabled'])
            elif n_denoising_bm4d.get() == 0:
                n_denoising_stv_chk.state(['!disabled'])

        def stv_clicked():
            if n_denoising_stv.get() == 1:
                n_denoising_bm4d_chk.state(['disabled'])
            elif n_denoising_stv.get() == 0:
                n_denoising_bm4d_chk.state(['!disabled'])

        # denoising method labels and dropdown list
        # tk.Label(self, text='Denoising method: ').place(x=10, y=95)

        n_denoising = tk.IntVar()
        n_denoising_chk = ttk.Checkbutton(self, text='Denoising', variable=n_denoising, command=denoising_clicked)
        n_denoising_chk.place(x=10, y=95)
        n_denoising.set(True)


        n_denoising_bm4d = tk.IntVar()
        n_denoising_bm4d_chk = ttk.Checkbutton(self, text='BM4d', variable=n_denoising_bm4d, command=bm4d_clicked)
        n_denoising_bm4d_chk.place(x=30, y=125)

        n_denoising_stv = tk.IntVar()
        n_denoising_stv_chk = ttk.Checkbutton(self, text='STV', variable=n_denoising_stv, command=stv_clicked)
        n_denoising_stv_chk.place(x=30, y=145)






        """
        method = ('Bm4d', 'STV')
        n_denoising = tk.StringVar()
        n_denoising_chosen = ttk.Combobox(self, width=8, textvariable=n_denoising)
        n_denoising_chosen['values'] = method
        n_denoising_chosen.set('Bm4d')
        n_denoising_chosen.place(x=130, y=95)
        """
        def decom_clicked():
            if n_decom.get() == 1:
                n_decom_mcr_chk.state(['!disabled'])
                n_decom_ls_chk.state(['!disabled'])
                n_decom_sp_chk.state(['!disabled'])
            elif n_decom.get() == 0:
                n_decom_mcr_chk.state(['disabled'])
                n_decom_ls_chk.state(['disabled'])
                n_decom_sp_chk.state(['disabled'])

        def sp_clicked():
            if n_decom_sp.get() == 1:
                n_decom_mcr_chk.state(['disabled'])
                n_decom_ls_chk.state(['disabled'])
            elif n_decom_sp.get() == 0:
                n_decom_mcr_chk.state(['!disabled'])
                n_decom_ls_chk.state(['!disabled'])

        def mcr_clicked():
            if n_decom_mcr.get() == 1:
                n_decom_sp_chk.state(['disabled'])
                n_decom_ls_chk.state(['disabled'])
            elif n_decom_mcr.get() == 0:
                n_decom_sp_chk.state(['!disabled'])
                n_decom_ls_chk.state(['!disabled'])

        def ls_clicked():
            if n_decom_ls.get() == 1:
                n_decom_sp_chk.state(['disabled'])
                n_decom_mcr_chk.state(['disabled'])
            elif n_decom_ls.get() == 0:
                n_decom_sp_chk.state(['!disabled'])
                n_decom_mcr_chk.state(['!disabled'])

        # decomposition method labels and dropdown list
        # tk.Label(self, text='Decomposition method: ').place(x=10, y=220)

        n_decom = tk.IntVar()
        n_decom_chk = ttk.Checkbutton(self, text='Decomposition', variable=n_decom, command=decom_clicked)
        n_decom_chk.place(x=10, y=220)
        n_decom.set(True)


        n_decom_sp = tk.IntVar()
        n_decom_sp_chk = ttk.Checkbutton(self, text='Spectral Phasor', variable=n_decom_sp, command=sp_clicked)
        n_decom_sp_chk.place(x=30, y=250)

        n_decom_mcr = tk.IntVar()
        n_decom_mcr_chk = ttk.Checkbutton(self, text='MCR', variable=n_decom_mcr, command=mcr_clicked)
        n_decom_mcr_chk.place(x=30, y=270)

        n_decom_ls = tk.IntVar()
        n_decom_ls_chk = ttk.Checkbutton(self, text='LS', variable=n_decom_ls, command=ls_clicked)
        n_decom_ls_chk.place(x=30, y=290)
        """
        decom = ('Spectral Phasor','MCR', 'LS')
        n_decomposition= tk.StringVar()
        n_decomposition_chosen = ttk.Combobox(self, width=8, textvariable=n_decomposition)
        n_decomposition_chosen['values'] = decom
        n_decomposition_chosen.set('MCR')
        n_decomposition_chosen.place(x=430, y=95)
        """



        # Separator
        separator1 = ttk.Separator(self, orient='horizontal')
        separator1.place(x=0, y=350, relwidth=1, relheight=1)


        # Next
        tk.Button(self, text='Next', command=Next).place(x=465, y=363)
        tk.Button(self, text='User Manual', command=master.user_manual).place(x=10, y=363)


class denoising_bm4d(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, height=800, width=800)  # need to define height and width to use .place()
        self.master.geometry('680x400')
        self.master.title('Chemical Image Analysis - Denoising - bm4d')
        global n
        n = 0
        print(batch)
        newpath = r'./denoise_bm4d'
        if not os.path.exists(newpath):
            os.mkdir(newpath)

        # img position
        img_x = 250
        img_y = 70

        def denoise_bm4d():

            global n
            n = 1
            """
            load = Image.open('gray.png')



            #width, height = load.size
            # ratio = 0.05
            load = load.resize((500, 250), Image.ANTIALIAS)
            render2 = ImageTk.PhotoImage(load)
            img2 = tk.Label(self, image=render2)
            img2.image = render2
            img2.place(x=200, y=50)
            """

            if n_estimated_sigma.get() == 'True':
                estimate_sigma = 1
            else:
                estimate_sigma = 0

            if n_wiener.get() == 'True':
                do_wiener = 1
            else:
                do_wiener = 0

            if n_verbose.get() == 'True':
                verbose = 1
            else:
                verbose = 0

            if n_variable_noise.get() == 'True':
                variable_noise = 1
            else:
                variable_noise = 0

            if filetype == '.tif':
                if tifffile.imread(filepath).dtype == 'uint16':
                    input_type = 'uint16'
                elif tifffile.imread(filepath).dtype == 'float32':
                    input_type = 'float32'
                elif tifffile.imread(filepath).dtype == 'uint8':
                    input_type = 'xxx'
            elif filetype == '.txt':
                input_type = 'txt'

            eng = matlab.engine.start_matlab()

            if batch == 0:
            # read original phantom
                if filetype == '.tif':
                    y = eng.im2double(eng.loadtiff(filepath))
                if filetype == '.txt':
                    y = eng.open_reshape_txt(filepath)
                if n_crop_phantom.get() == 'True':
                    y = eng.cropdata(y, 51, 125)

                print('Denoising Started')
                [PSNR, SSIM] = eng.bm4d_denoise_w_sigma(y, K, float(sigma_ent.get()), estimate_sigma, n_distribution.get(), n_profile.get(), do_wiener, verbose, variable_noise, int(noise_factor_ent.get()), input_type, filename, nargout=2)
                print(PSNR, SSIM)
            elif batch == 1:
                for i in range(len(os.listdir(directory))):
                    print(i)
                    filetype_batch = os.path.splitext(os.listdir(directory)[i])[1]
                    filename_batch = os.path.splitext(os.listdir(directory)[i])[0]
                    if filetype_batch == '.txt' or '.tif':
                        filepath_batch = directory + '/' + os.listdir(directory)[i]
                        if filetype_batch == '.tif':
                            y = eng.im2double(eng.loadtiff(filepath_batch))
                        if filetype_batch == '.txt':
                            y = eng.open_reshape_txt(filepath_batch)
                        if n_crop_phantom.get() == 'True':
                            y = eng.cropdata(y, 51, 125)

                        print('Denoising Started')
                        [PSNR, SSIM] = eng.bm4d_denoise_w_sigma(y, K, float(sigma_ent.get()), estimate_sigma,
                                                                n_distribution.get(), n_profile.get(), do_wiener,
                                                                verbose, variable_noise, int(noise_factor_ent.get()),
                                                                input_type, filename_batch, nargout=2)
                        print(PSNR, SSIM)

            eng.quit()

            def sel(self):
                if filetype == '.tif':
                    img = tifffile.imread(filepath)
                    if len(img.shape) == 2:
                        dst = np.zeros(shape=(img.shape[0], img.shape[1]))
                    elif len(img.shape) == 3:
                        dst = np.zeros(shape=(img.shape[1], img.shape[2]))

                    data_raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)

                    if len(img.shape) == 2:
                        load_raw = Image.fromarray(data_raw)
                    elif len(img.shape) == 3:
                        load_raw = Image.fromarray(data_raw[int(var.get())])
                elif filetype == '.txt':
                    img = tifffile.imread(filename + '.tif')
                    dst = np.zeros(shape=(M, N))
                    data_raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
                    load_raw = Image.fromarray(data_raw[int(var.get())])

                width, height = load_raw.size
                ratio = M / 200
                load_raw = load_raw.resize((int(width / ratio), int(height / ratio)), Image.NEAREST)
                render_raw = ImageTk.PhotoImage(load_raw)
                img_raw.configure(image=render_raw)
                img_raw.image = render_raw
                img_raw.place(x=220, y=60)


                data_denoise = tifffile.imread('./denoise_bm4d/' + filename + '_bm4d.tif')
                data_denoise = cv2.normalize(data_denoise, dst, 0, 255, cv2.NORM_MINMAX)

                if filetype == '.txt':
                    load_denoise = Image.fromarray(data_denoise[int(var.get())])
                elif filetype == '.tif':
                    img = tifffile.imread(filepath)
                    if len(img.shape) == 2:
                        load_denoise = Image.fromarray(data_denoise)
                    elif len(img.shape) == 3:
                        load_denoise = Image.fromarray(data_denoise[int(var.get())])

                width, height = load_denoise.size
                ratio = M / 200
                load_denoise = load_denoise.resize((int(width / ratio), int(height / ratio)), Image.NEAREST)
                render_denoise = ImageTk.PhotoImage(load_denoise)
                img_denoise.configure(image=render_denoise)
                img_denoise.image = render_denoise
                img_denoise.place(x=450, y=60)

            global var
            var = tk.DoubleVar(self)
            tk.Scale(self, variable=var, orient=tk.HORIZONTAL, from_=0, to=K - 1, length=370,
                     command=sel).place(x=220, y=285)
            var.set(int(K / 2))

            if filetype == '.tif':
                img = tifffile.imread(filepath)
                if len(img.shape) == 2:
                    dst = np.zeros(shape=(img.shape[0], img.shape[1]))
                elif len(img.shape) == 3:
                    dst = np.zeros(shape=(img.shape[1], img.shape[2]))
                data_raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)

                if len(img.shape) == 2:
                    load_raw = Image.fromarray(data_raw)
                elif len(img.shape) == 3:
                    load_raw = Image.fromarray(data_raw[int(K / 2)])
            elif filetype == '.txt':
                img = tifffile.imread(filename + '.tif')
                dst = np.zeros(shape=(M, N))
                data_raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
                load_raw = Image.fromarray(data_raw[int(K / 2)])

            width, height = load_raw.size
            ratio = M / 200
            load_raw = load_raw.resize((int(width / ratio), int(height / ratio)), Image.NEAREST)
            render_raw = ImageTk.PhotoImage(load_raw)
            global img_raw
            img_raw = tk.Label(self, image=render_raw)
            img_raw.image = render_raw
            img_raw.place(x=220, y=60)


            data_denoise = tifffile.imread('./denoise_bm4d/' + filename + '_bm4d.tif')
            data_denoise = cv2.normalize(data_denoise, dst, 0, 255, cv2.NORM_MINMAX)

            if filetype == '.txt':
                load_denoise = Image.fromarray(data_denoise[int(K / 2)])
            elif filetype == '.tif':
                if len(img.shape) == 2:
                    load_denoise = Image.fromarray(data_denoise)
                elif len(img.shape) == 3:
                    load_denoise = Image.fromarray(data_denoise[int(K / 2)])

            width, height = load_denoise.size
            ratio = M / 200
            load_denoise = load_denoise.resize((int(width / ratio), int(height / ratio)), Image.NEAREST)
            render_denoise = ImageTk.PhotoImage(load_denoise)
            # img_denoise.configure(image=render_denoise)
            global img_denoise
            img_denoise = tk.Label(self, image=render_denoise)
            img_denoise.image = render_denoise
            img_denoise.place(x=450, y=60)

            #
            tk.Label(self, text='Raw Image', font=(font, 10, 'bold')).place(x=280, y=33)
            tk.Label(self, text='Denoised Image', font=(font, 10, 'bold')).place(x=500, y=33)

            #
            def check():
                root = tk.Toplevel()
                root.geometry("300x90")
                root.title("Denoising Quality")
                x_position = 80
                tk.Label(root, text='PSNR: '+str(PSNR)).place(x=x_position, y=20)
                tk.Label(root, text='SSIM: '+str(SSIM)).place(x=x_position, y=50)

            tk.Button(self, text='Check Denoising Quality', command=check).place(x=150, y=365)

        def Next():
            if decomposition == True:
                if decomposition_method == 'MCR':
                    master.switch_frame(decomposition_mcr)
                elif decomposition_method == 'Spectral Phasor':
                    master.switch_frame(decomposition_phasor)
                elif decomposition_method == 'LS':
                    master.switch_frame(decomposition_ls)
            else:
                pass



        # Denoising label
        tk.Label(self, text='Denoising', font=(font, 14, 'bold')).place(x=10, y=10)

        # modifiable parameters - LABELS
        x_align = 10
        y_i = 55  # i for initial
        y_s = 30  # s for spacing
        tk.Label(self, text='Distribution:').place(x=x_align, y=y_i)
        tk.Label(self, text='Profile:').place(x=x_align, y=y_i+y_s)
        tk.Label(self, text='Do wiener:').place(x=x_align, y=y_i + 2*y_s)
        tk.Label(self, text='Verbose:').place(x=x_align, y=y_i + 3*y_s)
        tk.Label(self, text='Estimate Sigma:').place(x=x_align, y=y_i + 4*y_s)
        tk.Label(self, text='Crop Phantom:').place(x=x_align, y=y_i + 5*y_s)
        tk.Label(self, text='Variable noise:').place(x=x_align, y=y_i + 6*y_s)
        tk.Label(self, text='Sigma:').place(x=x_align, y=y_i + 7*y_s)
        tk.Label(self, text='Noise Factor:').place(x=x_align, y=y_i + 8*y_s)

        # modifiable parameters - ENTRIES
        x_s = 100
        sigma_ent = tk.Entry(self, width=13)
        noise_factor_ent = tk.Entry(self, width=13)
        sigma_ent.insert(0, '11')
        noise_factor_ent.insert(0, '3')
        sigma_ent.place(x=x_align+x_s, y=y_i + 7*y_s)
        noise_factor_ent.place(x=x_align+x_s, y=y_i + 8*y_s)

        # modifiable parameters - DROPDOWN LISTS
        n_distribution = tk.StringVar()
        n_distribution_chosen = ttk.Combobox(self, width=10, textvariable=n_distribution)
        n_distribution_chosen['values'] = ('Gauss', 'Rice')
        n_distribution_chosen.set('Gauss')
        n_distribution_chosen.place(x=x_align+x_s, y=y_i)

        n_profile = tk.StringVar()
        n_profile_chosen = ttk.Combobox(self, width=10, textvariable=n_profile)
        n_profile_chosen['values'] = ('mp', 'np', 'lc')
        n_profile_chosen.set('mp')
        n_profile_chosen.place(x=x_align+x_s, y=y_i+y_s)

        tf = ('True', 'False')
        n_wiener = tk.StringVar()
        n_wiener_chosen = ttk.Combobox(self, width=10, textvariable=n_wiener)
        n_wiener_chosen['values'] = tf
        n_wiener_chosen.set('True')
        n_wiener_chosen.place(x=x_align+x_s, y=y_i + 2*y_s)

        n_verbose = tk.StringVar()
        n_verbose_chosen = ttk.Combobox(self, width=10, textvariable=n_verbose)
        n_verbose_chosen['values'] = tf
        n_verbose_chosen.set('True')
        n_verbose_chosen.place(x=x_align+x_s, y=y_i + 3*y_s)

        n_estimated_sigma = tk.StringVar()
        n_estimated_sigma_chosen = ttk.Combobox(self, width=10, textvariable=n_estimated_sigma)
        n_estimated_sigma_chosen['values'] = tf
        n_estimated_sigma_chosen.set('True')
        n_estimated_sigma_chosen.place(x=x_align+x_s, y=y_i + 4*y_s)

        n_crop_phantom = tk.StringVar()
        n_crop_phantom_chosen = ttk.Combobox(self, width=10, textvariable=n_crop_phantom)
        n_crop_phantom_chosen['values'] = tf
        n_crop_phantom_chosen.set('False')
        n_crop_phantom_chosen.place(x=x_align+x_s, y=y_i + 5*y_s)

        n_variable_noise = tk.StringVar()
        n_variable_noise_chosen = ttk.Combobox(self, width=10, textvariable=n_variable_noise)
        n_variable_noise_chosen['values'] = tf
        n_variable_noise_chosen.set('False')
        n_variable_noise_chosen.place(x=x_align+x_s, y=y_i + 6*y_s)

        # Denoise button
        tk.Button(self, text='Denoise', command=denoise_bm4d).place(x=598, y=298)

        # Separator
        separator1 = ttk.Separator(self, orient='horizontal')
        separator1.place(x=0, y=350, relwidth=1, relheight=1)


        # Next and user manual
        tk.Button(self, text='Back', command=lambda: master.switch_frame(file_page)).place(x=10, y=365)
        tk.Button(self, text='User Manual', command=master.user_manual).place(x=60, y=365)
        if decomposition is True:
            tk.Button(self, text='Next', command=Next).place(x=630, y=365)
        else:
            tk.Button(self, text='Done', command=master.destroy).place(x=630, y=365)
            tk.Button(self, text='Export', command=master.export).place(x=570, y=365)



class denoising_stv(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, height=800, width=800)  # need to define height and width to use .place()
        self.master.geometry('680x400')
        self.master.title('Chemical Image Analysis - Denoising - STV')
        img_x = 250
        img_y = 70
        print(batch)
        print(directory)
        print(os.listdir(directory))
        global n
        n = 0
        newpath = r'./denoise_stv'
        if not os.path.exists(newpath):
            os.mkdir(newpath)

        def denoise_stv():
            global n
            n = 1

            eng = matlab.engine.start_matlab()
            eng.addpath(eng.genpath('./spectral_tv'))
            if filetype == '.tif':
                if tifffile.imread(filepath).dtype == 'uint16':
                    input_type = 'uint16'
                elif tifffile.imread(filepath).dtype == 'float32':
                    input_type = 'float32'
                elif tifffile.imread(filepath).dtype == 'uint8':
                    input_type = 'xxx'

            elif filetype == '.txt':
                input_type = 'txt'

            print('batch is ' + str(batch))
            if batch == 0:
                # read original phantom
                if filetype == '.tif':
                    hyper_noisy = eng.read_hyperdata(filepath, M, N, K)
                if filetype == '.txt':
                    hyper_noisy = eng.open_reshape_txt(filepath)

                beta = beta_ent.get().split(",")
                [PSNR, SSIM, out_stv_sigma] = eng.make_beta_array(n_tv_method.get(), float(rho_r_ent.get()), float(rho_o_ent.get()),
                                                   float(beta[0]), float(beta[1]), float(beta[2]),
                                                   float(gamma_ent.get()), float(max_itr_ent.get()),
                                                   float(alpha_ent.get()), float(tol_ent.get()), hyper_noisy, K,
                                                   input_type, filename, nargout=3)
                print(PSNR, SSIM)
            elif batch == 1:
                for i in range(len(os.listdir(directory))):
                    print(i)
                    filetype_batch = os.path.splitext(os.listdir(directory)[i])[1]
                    filename_batch = os.path.splitext(os.listdir(directory)[i])[0]

                    if filetype_batch == '.txt' or '.tif':
                        filepath_batch = directory + '/'+ os.listdir(directory)[i]
                        if filetype_batch == '.tif':
                            hyper_noisy = eng.read_hyperdata(filepath_batch, M, N, K)
                        if filetype_batch == '.txt':
                            hyper_noisy = eng.open_reshape_txt(filepath_batch)

                        if i == 0:
                            beta = beta_ent.get().split(",")
                            [PSNR, SSIM, out_stv_sigma] = eng.make_beta_array(n_tv_method.get(), float(rho_r_ent.get()),
                                                                        float(rho_o_ent.get()),
                                                                        float(beta[0]), float(beta[1]), float(beta[2]),
                                                                        float(gamma_ent.get()),
                                                                        float(max_itr_ent.get()),
                                                                        float(alpha_ent.get()), float(tol_ent.get()),
                                                                        hyper_noisy, K,
                                                                        input_type, filename, nargout=3)
                        else:
                            beta = beta_ent.get().split(",")
                            [PSNR, SSIM] = eng.make_beta_array_batch(out_stv_sigma, n_tv_method.get(), float(rho_r_ent.get()),
                                                                        float(rho_o_ent.get()),
                                                                        float(beta[0]), float(beta[1]), float(beta[2]),
                                                                        float(gamma_ent.get()),
                                                                        float(max_itr_ent.get()),
                                                                        float(alpha_ent.get()), float(tol_ent.get()),
                                                                        hyper_noisy, K,
                                                                        input_type, filename_batch, nargout=2)
                        print(PSNR, SSIM)

            eng.quit()

            def sel(self):
                if filetype == '.tif':
                    img = tifffile.imread(filepath)
                    if len(img.shape) == 2:
                        dst = np.zeros(shape=(img.shape[0], img.shape[1]))
                    elif len(img.shape) == 3:
                        dst = np.zeros(shape=(img.shape[1], img.shape[2]))

                    data_raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)

                    if len(img.shape) == 2:
                        load_raw = Image.fromarray(data_raw)
                    elif len(img.shape) == 3:
                        load_raw = Image.fromarray(data_raw[int(var.get())])
                elif filetype == '.txt':
                    img = tifffile.imread(filename+'.tif')
                    dst = np.zeros(shape=(M, N))
                    data_raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
                    load_raw = Image.fromarray(data_raw[int(var.get())])



                width, height = load_raw.size
                ratio = M/200
                load_raw = load_raw.resize((int(width / ratio), int(height / ratio)), Image.NEAREST)
                render_raw = ImageTk.PhotoImage(load_raw)
                img_raw.configure(image=render_raw)
                img_raw.image = render_raw
                img_raw.place(x=220, y=60)


                data_denoise = tifffile.imread('./denoise_stv/'+filename+'_stv.tif')
                data_denoise = cv2.normalize(data_denoise, dst, 0, 255, cv2.NORM_MINMAX)

                if filetype == '.txt':
                    load_denoise = Image.fromarray(data_denoise[int(var.get())])
                elif filetype == '.tif':
                    img = tifffile.imread(filepath)
                    if len(img.shape) == 2:
                        load_denoise = Image.fromarray(data_denoise)
                    elif len(img.shape) == 3:
                        load_denoise = Image.fromarray(data_denoise[int(var.get())])

                width, height = load_denoise.size
                ratio = M/200
                load_denoise = load_denoise.resize((int(width / ratio), int(height / ratio)), Image.NEAREST)
                render_denoise = ImageTk.PhotoImage(load_denoise)
                img_denoise.configure(image=render_denoise)
                img_denoise.image = render_denoise
                img_denoise.place(x=450, y=60)

            global var
            var = tk.DoubleVar(self)
            tk.Scale(self, variable=var, orient=tk.HORIZONTAL, from_=0, to=K-1, length=370,
                     command=sel).place(x=220, y=285)
            var.set(int(K/2))

            if filetype == '.tif':
                img = tifffile.imread(filepath)
                if len(img.shape) == 2:
                    dst = np.zeros(shape=(img.shape[0], img.shape[1]))
                elif len(img.shape) == 3:
                    dst = np.zeros(shape=(img.shape[1], img.shape[2]))
                data_raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)

                if len(img.shape) == 2:
                    load_raw = Image.fromarray(data_raw)
                elif len(img.shape) == 3:
                    load_raw = Image.fromarray(data_raw[int(K / 2)])
            elif filetype == '.txt':
                img = tifffile.imread(filename + '.tif')
                dst = np.zeros(shape=(M, N))
                data_raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
                load_raw = Image.fromarray(data_raw[int(K / 2)])

            width, height = load_raw.size
            ratio = M/200
            load_raw = load_raw.resize((int(width / ratio), int(height / ratio)), Image.NEAREST)
            render_raw = ImageTk.PhotoImage(load_raw)
            global img_raw
            img_raw = tk.Label(self, image=render_raw)
            img_raw.image = render_raw
            img_raw.place(x=220, y=60)

            data_denoise = tifffile.imread('./denoise_stv/'+filename+'_stv.tif')
            #if filetype == '.tif':
            data_denoise = cv2.normalize(data_denoise, dst, 0, 255, cv2.NORM_MINMAX)

            if filetype == '.txt':
                load_denoise = Image.fromarray(data_denoise[int(K/2)])
            elif filetype == '.tif':
                if len(img.shape) == 2:
                    load_denoise = Image.fromarray(data_denoise)
                elif len(img.shape) == 3:
                    load_denoise = Image.fromarray(data_denoise[int(K/2)])

            width, height = load_denoise.size
            ratio = M/200
            load_denoise = load_denoise.resize((int(width / ratio), int(height / ratio)), Image.NEAREST)
            render_denoise = ImageTk.PhotoImage(load_denoise)
            # img_denoise.configure(image=render_denoise)
            global img_denoise
            img_denoise = tk.Label(self, image=render_denoise)
            img_denoise.image = render_denoise
            img_denoise.place(x=450, y=60)

            #
            tk.Label(self, text='Raw Image', font=(font, 10, 'bold')).place(x=280, y=33)
            tk.Label(self, text='Denoised Image', font=(font, 10, 'bold')).place(x=500, y=33)

            #
            def check():
                root = tk.Toplevel()
                root.geometry("300x90")
                root.title("Denoising Quality")
                x_position = 80
                tk.Label(root, text='PSNR: '+str(PSNR)).place(x=x_position, y=20)
                tk.Label(root, text='SSIM: '+str(SSIM)).place(x=x_position, y=50)


            tk.Button(self, text='Check Denoising Quality', command=check).place(x=150, y=365)

        def Next():
            if decomposition_method == 'MCR':
                master.switch_frame(decomposition_mcr)
            elif decomposition_method == 'Spectral Phasor':
                master.switch_frame(decomposition_phasor)
            elif decomposition_method == 'LS':
                master.switch_frame(decomposition_ls)

        # Denoising label
        tk.Label(self, text='Denoising', font=(font, 14, 'bold')).place(x=10, y=10)

        # Denoise button
        tk.Button(self, text='Denoise', command=denoise_stv).place(x=598, y=298)

        # modifiable parameters - LABELS
        x_align = 10
        y_i = 55  # i for initial
        y_s = 30  # s for spacing
        tk.Label(self, text='tv method:').place(x=x_align, y=y_i)
        tk.Label(self, text='rho_r:').place(x=x_align, y=y_i+y_s)
        tk.Label(self, text='rho_o:').place(x=x_align, y=y_i + 2*y_s)
        tk.Label(self, text='beta:').place(x=x_align, y=y_i + 3*y_s)
        tk.Label(self, text='gamma:').place(x=x_align, y=y_i + 4*y_s)
        tk.Label(self, text='max iteration:').place(x=x_align, y=y_i + 5*y_s)
        tk.Label(self, text='alpha:').place(x=x_align, y=y_i + 6*y_s)
        tk.Label(self, text='tolerance:').place(x=x_align, y=y_i + 7*y_s)
        # tk.Label(self, text='f:').place(x=x_align, y=y_i + 8*y_s)

        # modifiable parameters - ENTRIES
        x_s = 100

        # modifiable parameters - DROPDOWN LISTS
        n_tv_method = tk.StringVar()
        n_tv_method_chosen = ttk.Combobox(self, width=10, textvariable=n_tv_method)
        n_tv_method_chosen['values'] = ('aniso', 'iso')
        n_tv_method_chosen.set('aniso')
        n_tv_method_chosen.place(x=x_align+x_s, y=y_i)

        rho_r_ent = tk.Entry(self, width=13)
        rho_r_ent.insert(0, '2')
        rho_r_ent.place(x=x_align+x_s, y=y_i + y_s)

        rho_o_ent = tk.Entry(self, width=13)
        rho_o_ent.insert(0, '50')
        rho_o_ent.place(x=x_align+x_s, y=y_i + 2*y_s)

        beta_ent = tk.Entry(self, width=13)
        beta_ent.insert(0, '1,1,0')
        beta_ent.place(x=x_align+x_s, y=y_i + 3*y_s)

        gamma_ent = tk.Entry(self, width=13)
        gamma_ent.insert(0, '2')
        gamma_ent.place(x=x_align+x_s, y=y_i + 4*y_s)

        max_itr_ent = tk.Entry(self, width=13)
        max_itr_ent.insert(0, '20')
        max_itr_ent.place(x=x_align+x_s, y=y_i + 5*y_s)

        alpha_ent = tk.Entry(self, width=13)
        alpha_ent.insert(0, '0.7')
        alpha_ent.place(x=x_align+x_s, y=y_i + 6*y_s)

        tol_ent = tk.Entry(self, width=13)
        tol_ent.insert(0, '1e-3')
        tol_ent.place(x=x_align+x_s, y=y_i + 7*y_s)

        # Separator
        separator1 = ttk.Separator(self, orient='horizontal')
        separator1.place(x=0, y=350, relwidth=1, relheight=1)

        # Next and user manual
        tk.Button(self, text='Back', command=lambda: master.switch_frame(file_page)).place(x=10, y=365)
        tk.Button(self, text='User Manual', command=master.user_manual).place(x=60, y=365)
        if decomposition is True:
            tk.Button(self, text='Next', command=Next).place(x=630, y=365)
        else:
            tk.Button(self, text='Done', command=master.destroy).place(x=630, y=365)
            tk.Button(self, text='Export', command=master.export).place(x=570, y=365)



class decomposition_mcr(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, height=800, width=800)  # need to define height and width to use .place()
        self.master.geometry('680x400')
        self.master.title('Chemical Image Analysis - Decomposition - MCR')

        def back():
            if denoising is not True:
                master.switch_frame(file_page)
            else:
                if denoising_method == 'Bm4d':
                    master.switch_frame(denoising_bm4d)
                elif denoising_method == 'STV':
                    master.switch_frame(denoising_stv)

        def open_file():
            """Open a file for editing"""
            global varlist
            if ent_lbl.get() == '':
                global spectralpath
                spectralpath = askopenfilename(
                    filetypes=[("MAT File", "*.mat")]
                )
                if not spectralpath:
                    return
                ent_lbl_text.set(spectralpath)
            else:
                spectralpath = ent_lbl.get()

            eng = matlab.engine.start_matlab()
            varlist = eng.who(eng.matfile(spectralpath))
            eng.quit()
            n_component_chosen['values'] = varlist
            n_component_chosen.set(varlist[0])
            component_lbl['text'] = 'Components (' + str(len(varlist)) + '): '

        def decompose():
            eng = matlab.engine.start_matlab()
            if batch == 0:
                if denoising is False or n_use.get() == 0:
                    input_file = filepath
                else:
                    if denoising_method == 'Bm4d':
                        if filetype == '.tif':
                            input_file = './denoise_bm4d/' + filename + '_bm4d.tif'
                        elif filetype == '.txt':
                            input_file = './denoise_bm4d/' + filename + '_bm4d_32bit.tif'
                    elif denoising_method == 'STV':
                        if filetype == '.tif':
                            input_file = './denoise_stv/'+filename+'_stv.tif'
                        elif filetype == '.txt':
                            input_file = './denoise_stv/'+filename+'_stv_32bit.tif'
                eng.mcr(input_file, filename, spectralpath, float(peaks_ent.get().split(',')[0]),
                             float(peaks_ent.get().split(',')[1]), float(shifts_ent.get().split(',')[0]),
                             float(shifts_ent.get().split(',')[1]), float(level_ent.get()), float(aug_ent.get()), float(itr_ent.get()), nargout=0)



            eng.quit()

            global img_ls
            img = tifffile.imread('./mcr_chemical_maps/' + filename + '_' + n_component.get() + '.tif')
            load_raw = Image.fromarray(img)
            r = 1.1
            load_raw = load_raw.resize((int(400/r), int(300/r)), Image.ANTIALIAS)
            render_raw = ImageTk.PhotoImage(load_raw)
            img_ls = tk.Label(self, image=render_raw)
            img_ls.image = render_raw
            img_ls.place(x=70, y=70)

            # View spectra button
            def spectra():
                root = tk.Toplevel()
                root.geometry("450x350")
                root.title("Chemical Spectra")
                image = Image.open('./mcr_chemical_maps/pure_chemical_spectra.png')
                width, height = image.size
                load = image.resize((int(width / 2), int(height / 2)), Image.ANTIALIAS)
                render = ImageTk.PhotoImage(load)
                img = tk.Label(root, image=render)
                img.image = render
                img.place(x=10, y=20)

                # modifiable parameters - DROPDOWN LISTS
                def spectrum_change(self):
                    if n_spectrum.get() == 'Original':
                        image = Image.open('./mcr_chemical_maps/pure_chemical_spectra.png')
                    if n_spectrum.get() == 'New':
                        image = Image.open('./mcr_chemical_maps/new_spectral_profiles.png')
                    width, height = image.size
                    load = image.resize((int(width / 2), int(height / 2)), Image.ANTIALIAS)
                    render = ImageTk.PhotoImage(load)
                    img.configure(image=render)
                    img.image = render
                    img.place(x=10, y=20)

                spectrum_lbl = tk.Label(root, text='Spectra:')
                spectrum_lbl.place(x=330, y=10)
                n_spectrum = tk.StringVar()
                n_spectrum_chosen = ttk.Combobox(root, width=7, textvariable=n_spectrum)
                n_spectrum_chosen.bind("<<ComboboxSelected>>", spectrum_change)
                n_spectrum_chosen['values'] = ('Original', 'New')
                n_spectrum_chosen.set('Original')
                n_spectrum_chosen.place(x=380, y=10)

            tk.Button(self, text='View Spectra', command=spectra).place(x=150, y=365)

        def switch(self):  # self is needed here
            img = tifffile.imread('./mcr_chemical_maps/' + filename + '_' + n_component.get() + '.tif')
            load_raw = Image.fromarray(img)
            r = 1.1
            load_raw = load_raw.resize((int(400/r), int(300/r)), Image.ANTIALIAS)
            render_raw = ImageTk.PhotoImage(load_raw)
            img_ls.configure(image=render_raw)
            img_ls.image = render_raw
            img_ls.place(x=70, y=70)


        # Decomposition label
        tk.Label(self, text='Decomposition', font=(font, 14, 'bold')).place(x=10, y=10)

        # spectral profile
        tk.Label(self, text='Spectral Profile:').place(x=10, y=45)
        ent_lbl_text = tk.StringVar()
        ent_lbl_text.set('')
        ent_lbl = tk.Entry(self, width=60, textvariable=ent_lbl_text)
        ent_lbl.place(x=110, y=47)

        # use denoised image ???
        n_use = tk.IntVar()
        n_use_chk = ttk.Checkbutton(self, text='Use Denoised Image', variable=n_use)
        n_use_chk.place(x=540, y=45)
        if denoising is False:
            n_use_chk.state(['disabled'])
            # n_use.set(False)
        else:
            n_use.set(True)

        # load button
        tk.Button(self, text='Load', command=open_file).place(x=490, y=42)

        # entries
        tk.Label(self, text='Peaks:').place(x=490, y=90)
        tk.Label(self, text='Raman Shifts:').place(x=490, y=120)
        tk.Label(self, text='Sparsity Level:').place(x=490, y=150)
        tk.Label(self, text='Augmentation:').place(x=490, y=180)
        tk.Label(self, text='Iteration:').place(x=490, y=210)
        peaks_ent = tk.Entry(self, width=10)
        peaks_ent.place(x=590, y=90)
        peaks_ent.insert(0, '40,76')
        shifts_ent = tk.Entry(self, width=10)
        shifts_ent.place(x=590, y=120)
        shifts_ent.insert(0, '2913,2994')
        level_ent = tk.Entry(self, width=10)
        level_ent.place(x=590, y=150)
        level_ent.insert(0, '5e-2')
        aug_ent = tk.Entry(self, width=10)
        aug_ent.place(x=590, y=180)
        aug_ent.insert(0, '0.5')
        itr_ent = tk.Entry(self, width=10)
        itr_ent.place(x=590, y=210)
        itr_ent.insert(0, '5')

        # modifiable parameters - DROPDOWN LISTS
        component_lbl = tk.Label(self, text='Components:')
        component_lbl.place(x=490, y=240)
        n_component = tk.StringVar()
        n_component_chosen = ttk.Combobox(self, width=7, textvariable=n_component)
        n_component_chosen.bind("<<ComboboxSelected>>", switch)
        n_component_chosen.set('')
        n_component_chosen.place(x=590, y=240)

        # Decompose button
        tk.Button(self, text='Decompose', command=decompose).place(x=591, y=310)

        # Separator
        separator1 = ttk.Separator(self, orient='horizontal')
        separator1.place(x=0, y=350, relwidth=1, relheight=1)

        tk.Button(self, text='Back', command=back).place(x=10, y=365)
        tk.Button(self, text='User Manual', command=master.user_manual).place(x=60, y=365)
        tk.Button(self, text='Done', command=master.destroy).place(x=630, y=365)
        tk.Button(self, text='Export', command=master.export).place(x=570, y=365)



class decomposition_phasor(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, height=800, width=800)  # need to define height and width to use .place()
        self.master.geometry('680x400')
        self.master.title('Chemical Image Analysis - Decomposition - Spectral Phasor')

        def back():
            if denoising is not True:
                master.switch_frame(file_page)
            else:
                if denoising_method == 'Bm4d':
                    master.switch_frame(denoising_bm4d)
                elif denoising_method == 'STV':
                    master.switch_frame(denoising_stv)

        # Decomposition label
        tk.Label(self, text='Decomposition', font=(font, 14, 'bold')).place(x=10, y=10)

        # Separator
        separator1 = ttk.Separator(self, orient='horizontal')
        separator1.place(x=0, y=350, relwidth=1, relheight=1)

        tk.Button(self, text='Back', command=back).place(x=10, y=365)
        tk.Button(self, text='User Manual', command=master.user_manual).place(x=60, y=365)
        tk.Button(self, text='Done', command=master.destroy).place(x=630, y=365)


class decomposition_ls(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, height=800, width=800)  # need to define height and width to use .place()
        self.master.geometry('680x400')
        self.master.title('Chemical Image Analysis - Decomposition - Least Square Fitting')


        def back():
            if denoising is not True:
                master.switch_frame(file_page)
            else:
                if denoising_method == 'Bm4d':
                    master.switch_frame(denoising_bm4d)
                elif denoising_method == 'STV':
                    master.switch_frame(denoising_stv)

        def open_file():
            """Open a file for editing"""
            global varlist
            if ent_lbl.get() == '':
                global spectralpath
                spectralpath = askopenfilename(
                    filetypes=[("MAT File", "*.mat")]
                )
                if not spectralpath:
                    return
                ent_lbl_text.set(spectralpath)
            else:
                spectralpath = ent_lbl.get()

            eng = matlab.engine.start_matlab()
            varlist = eng.who(eng.matfile(spectralpath))
            eng.quit()
            n_component_chosen['values'] = varlist
            n_component_chosen.set(varlist[0])
            component_lbl['text'] = 'Components (' + str(len(varlist)) + '): '

        def decompose():
            eng = matlab.engine.start_matlab()
            print(n_normalize.get())
            if batch == 0:
                if denoising is False or n_use.get()==0:
                    input_file = filepath
                else:
                    if denoising_method == 'Bm4d':
                        if filetype == '.tif':
                            input_file = './denoise_bm4d/' + filename + '_bm4d.tif'
                        elif filetype == '.txt':
                            input_file = './denoise_bm4d/' + filename + '_bm4d_32bit.tif'
                    elif denoising_method == 'STV':
                        if filetype == '.tif':
                            input_file = './denoise_stv/'+filename+'_stv.tif'
                        elif filetype == '.txt':
                            input_file = './denoise_stv/'+filename+'_stv_32bit.tif'
                eng.least_square(input_file, filename, spectralpath, n_normalize.get(),float(peaks_ent.get().split(',')[0]), float(peaks_ent.get().split(',')[1]), float(shifts_ent.get().split(',')[0]), float(shifts_ent.get().split(',')[1]), float(level_ent.get()), nargout=0)
            elif batch == 1:
                global filename_batch
                if denoising is False or n_use.get() == 0:
                    count = 0
                    for i in range(len(os.listdir(directory))):
                        print(i)
                        filetype_batch = os.path.splitext(os.listdir(directory)[i])[1]
                        filename_batch = os.path.splitext(os.listdir(directory)[i])[0]

                        if filetype_batch == '.txt' or '.tif':
                            count = count + 1
                            filepath_batch = directory + '/' + os.listdir(directory)[i]
                            if count == 1:
                                sigma = eng.least_square(filepath_batch, filename_batch, spectralpath, n_normalize.get(),
                                                 float(peaks_ent.get().split(',')[0]),
                                                 float(peaks_ent.get().split(',')[1]),
                                                 float(shifts_ent.get().split(',')[0]),
                                                 float(shifts_ent.get().split(',')[1]), float(level_ent.get()),
                                                 nargout=1)
                            else:
                                eng.least_square_batch(sigma, filepath_batch, filename_batch, spectralpath, n_normalize.get(),
                                                 float(peaks_ent.get().split(',')[0]),
                                                 float(peaks_ent.get().split(',')[1]),
                                                 float(shifts_ent.get().split(',')[0]),
                                                 float(shifts_ent.get().split(',')[1]), float(level_ent.get()),
                                                 nargout=0)
                else:
                    if denoising_method == 'Bm4d':
                        count = 0
                        for i in range(len(os.listdir('./denoise_bm4d'))):
                            filetype_batch = os.path.splitext(os.listdir('./denoise_bm4d')[i])[1]
                            filename_batch = os.path.splitext(os.listdir('./denoise_bm4d')[i])[0]

                            if filename_batch != '.png':
                                count = count + 1
                                filepath_batch = './denoise_bm4d/' + os.listdir('./denoise_bm4d')[i]
                                if count == 1:
                                    sigma = eng.least_square(filepath_batch, filename_batch, spectralpath,
                                                             n_normalize.get(),
                                                             float(peaks_ent.get().split(',')[0]),
                                                             float(peaks_ent.get().split(',')[1]),
                                                             float(shifts_ent.get().split(',')[0]),
                                                             float(shifts_ent.get().split(',')[1]),
                                                             float(level_ent.get()),
                                                             nargout=1)
                                else:
                                    eng.least_square_batch(sigma, filepath_batch, filename_batch, spectralpath,
                                                           n_normalize.get(),
                                                           float(peaks_ent.get().split(',')[0]),
                                                           float(peaks_ent.get().split(',')[1]),
                                                           float(shifts_ent.get().split(',')[0]),
                                                           float(shifts_ent.get().split(',')[1]),
                                                           float(level_ent.get()),
                                                           nargout=0)
                    elif denoising_method == 'STV':
                        count = 0
                        for i in range(len(os.listdir('./denoise_stv'))):
                            filetype_batch = os.path.splitext(os.listdir('./denoise_stv')[i])[1]
                            filename_batch = os.path.splitext(os.listdir('./denoise_stv')[i])[0]
                            filename_final = filename_batch.split('_stv')[0]

                            if filetype_batch != '.png':
                                count = count + 1
                                filepath_batch = './denoise_stv/' + os.listdir('./denoise_stv')[i]
                                if count == 1:
                                    sigma = eng.least_square(filepath_batch, filename_final, spectralpath,
                                                             n_normalize.get(),
                                                             float(peaks_ent.get().split(',')[0]),
                                                             float(peaks_ent.get().split(',')[1]),
                                                             float(shifts_ent.get().split(',')[0]),
                                                             float(shifts_ent.get().split(',')[1]),
                                                             float(level_ent.get()),
                                                             nargout=1)
                                else:
                                    eng.least_square_batch(sigma, filepath_batch, filename_final, spectralpath,
                                                           n_normalize.get(),
                                                           float(peaks_ent.get().split(',')[0]),
                                                           float(peaks_ent.get().split(',')[1]),
                                                           float(shifts_ent.get().split(',')[0]),
                                                           float(shifts_ent.get().split(',')[1]),
                                                           float(level_ent.get()),
                                                           nargout=0)

            eng.quit()

            # View spectra button
            def spectra():
                root = tk.Toplevel()
                root.geometry("450x350")
                root.title("Chemical Spectra")
                image = Image.open('./ls_chemical_maps/pure_chemical_spectra.png')
                width, height = image.size
                load = image.resize((int(width/ 2), int(height/ 2)), Image.ANTIALIAS)
                render = ImageTk.PhotoImage(load)
                img = tk.Label(root, image=render)
                img.image = render
                img.place(x=10, y=20)

            tk.Button(self, text='View Spectra', command=spectra).place(x=150, y=365)
            global img_ls
            img = tifffile.imread('./ls_chemical_maps/'+ filename + '_' + n_component.get()+'.tif')

            #dst = np.zeros(shape=(M, N))
            #img = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
            load_raw = Image.fromarray(img)
            r = 1.1
            load_raw = load_raw.resize((int(400/r), int(300/r)), Image.ANTIALIAS)
            render_raw = ImageTk.PhotoImage(load_raw)
            img_ls = tk.Label(self, image=render_raw)
            img_ls.image = render_raw
            img_ls.place(x=70, y=70)

        def switch(self): # self is needed here
            img = tifffile.imread('./ls_chemical_maps/'+ filename + '_' + n_component.get()+'.tif')
            #dst = np.zeros(shape=(M, N))
            #img = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
            load_raw = Image.fromarray(img)
            r = 1.1
            load_raw = load_raw.resize((int(400/r), int(300/r)), Image.ANTIALIAS)
            render_raw = ImageTk.PhotoImage(load_raw)
            img_ls.configure(image=render_raw)
            img_ls.image = render_raw
            img_ls.place(x=70, y=70)

        # Decomposition label
        tk.Label(self, text='Decomposition', font=(font, 14, 'bold')).place(x=10, y=10)

        # spectral profile
        tk.Label(self, text='Spectral Profile:').place(x=10, y=45)
        ent_lbl_text = tk.StringVar()
        ent_lbl_text.set('')
        ent_lbl = tk.Entry(self, width=60, textvariable=ent_lbl_text)
        ent_lbl.place(x=110, y=47)

        # use denoised image ???
        n_use = tk.IntVar()
        n_use_chk = ttk.Checkbutton(self, text='Use Denoised Image', variable=n_use)
        n_use_chk.place(x=540, y=35)
        if denoising is False:
            n_use_chk.state(['disabled'])
            # n_use.set(False)
        else:
            n_use.set(True)

        # normalize image ???
        n_normalize = tk.IntVar()
        n_normalize_chk = ttk.Checkbutton(self, text='Normalize Image', variable=n_normalize)
        n_normalize_chk.place(x=540, y=55)
        n_normalize.set(True)


        # load button
        tk.Button(self, text='Load', command=open_file).place(x=490, y=42)

        # entries
        tk.Label(self, text='Peaks:').place(x=490, y=90)
        tk.Label(self, text='Raman Shifts:').place(x=490, y=120)
        tk.Label(self, text='Sparsity Level:').place(x=490, y=150)
        peaks_ent = tk.Entry(self, width=10)
        peaks_ent.place(x=590, y=90)
        peaks_ent.insert(0,'40,76')
        shifts_ent = tk.Entry(self, width=10)
        shifts_ent.place(x=590, y=120)
        shifts_ent.insert(0,'2913,2994')
        level_ent = tk.Entry(self, width=10)
        level_ent.place(x=590, y=150)
        level_ent.insert(0, '5e-2')

        # modifiable parameters - DROPDOWN LISTS
        component_lbl = tk.Label(self, text='Components:')
        component_lbl.place(x=490, y=180)
        n_component = tk.StringVar()
        n_component_chosen = ttk.Combobox(self, width=7, textvariable=n_component)
        n_component_chosen.bind("<<ComboboxSelected>>", switch)
        n_component_chosen.set('')
        n_component_chosen.place(x=590, y=180)

        # Decompose button
        tk.Button(self, text='Decompose', command=decompose).place(x=591, y=310)

        # Separator
        separator1 = ttk.Separator(self, orient='horizontal')
        separator1.place(x=0, y=350, relwidth=1, relheight=1)

        tk.Button(self, text='Back', command=back).place(x=10, y=365)
        tk.Button(self, text='User Manual', command=master.user_manual).place(x=60, y=365)
        tk.Button(self, text='Done', command=master.destroy).place(x=630, y=365)
        tk.Button(self, text='Export', command=master.export).place(x=570, y=365)


if __name__ == "__main__":
    app = Main()
    #app.title("Denoising v0.1")
    app.mainloop()


