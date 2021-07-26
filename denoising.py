import tkinter as tk
from tkinter import ttk, Canvas
import os
from tkinter.filedialog import askopenfilename
import matlab.engine
import tifffile
from PIL import Image, ImageTk
import cv2
import webbrowser
import numpy as np
import threading
import time
from tqdm import tqdm
#from PIL.Image import core as _imaging
import progressbar as pb


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


class file_page(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, height=800, width=800)  # need to define height and width to use .place()
        self.master.geometry('520x400')
        self.master.title('Chemical Image Analysis')
        img_x = 280
        img_y = 95
        try:
            os.path.exists('test.tif')
        except ValueError:
            pass

        def open_file():
            """Open a file for editing"""

            if ent_lbl.get() == '':
                global filepath
                filepath = askopenfilename(
                    filetypes=[("Tif File", "*.tif"), ("Txt File", "*.txt"), ("All Files", "*.*")]
                )
                if not filepath:
                    return
                ent_lbl_text.set(filepath)
            else:
                filepath = ent_lbl.get()

            # global variables used later
            global directory
            global filetype
            global frame_number
            directory = os.path.dirname(os.path.abspath(filepath))
            filename_w_ext = os.path.basename(filepath)
            filetype = os.path.splitext(filename_w_ext)[1]
            eng = matlab.engine.start_matlab()

            if filetype == '.tif':
                if tifffile.imread(filepath).dtype == 'uint16':
                    img = tifffile.imread(filepath)
                    dst = np.zeros(shape=(img.shape[1], img.shape[2]))
                    raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
                    im = np.uint8(raw)
                    imlist = []
                    for m in im:
                        imlist.append(Image.fromarray(m))
                # imlist[0].save("test.tif", compression="tiff_deflate", save_all=True,
                #               append_images=imlist[1:])
                    imlist[0].save("test.tif", save_all=True, append_images=imlist[1:])

            global K, M, N
            if filetype == '.tif':
                A = eng.size(eng.imread(filepath))
                M = int(A[0][0])
                N = int(A[0][1])
                K = int(eng.get_frame(filepath))
                text = 'The dimensions of the file is ' + str(M) + ' x ' + str(N) + ' x ' + str(K) + '.'

                def sel(self):
                    # load_raw = Image.fromarray(data_raw[int(var.get())].astype('uint8'))
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
                global data_raw
                img = tifffile.imread(filepath)
                dst = np.zeros(shape=(img.shape[1], img.shape[2]))
                data_raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
                #data_raw = tifffile.imread(filepath)
                load_raw = Image.fromarray(data_raw[int(K / 2)])
                width, height = load_raw.size
                ratio = M / 200
                load_raw = load_raw.resize((int(width / ratio), int(height / ratio)), Image.ANTIALIAS)
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
                text = 'The dimensions of the file is ' + str(N) + ' x ' + str(M) + '. Click "Transform" to perform a 3D transformation.'
                tk.Button(self, text='Transform', command=transform).place(x=100, y=363)
            eng.quit()

            tk.Label(self, text=text).place(x=10, y=50)

        def transform():
            eng = matlab.engine.start_matlab()
            eng.txt_to_tif(filepath, nargout=0)
            eng.quit()

            def sel(self):
                # load_raw = Image.fromarray(data_raw[int(var.get())].astype('uint8'))
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
            data_raw = tifffile.imread('raw.tif')
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
            global denoising, decomposition, decomposition_method, denoising_method
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

        # filepath entry
        ent_lbl_text = tk.StringVar()
        ent_lbl_text.set('')
        ent_lbl = tk.Entry(self, width=68, textvariable=ent_lbl_text)
        ent_lbl.place(x=90, y=14)

        # File selection buttons
        tk.Button(self, text="Select File", command=open_file).place(x=10, y=10)

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
        n_denoising_bm4d_chk = ttk.Checkbutton(self, text='Bm4d', variable=n_denoising_bm4d, command=bm4d_clicked)
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

        # img position
        img_x = 250
        img_y = 70

        def denoise_bm4d():
            """
            def real():
                progress = ttk.Progressbar(self, orient='horizontal', length=100, mode='indeterminate')
                progress.place(x=20, y=400)
                progress.start()
                time.sleep(1000)
                progress.stop()
            threading.Thread(target=real, args=()).start()
            """
            global n
            n = 1
            load = Image.open('gray.png')
            """
            progress = ttk.Progressbar(self, orient="horizontal", length=100, mode='indeterminate')
            progress.place(x=10, y=400)
            progress.start()
            progress['maximum'] = 3
            progress["value"] = 1
            progress.update()
            """


            #width, height = load.size
            # ratio = 0.05
            load = load.resize((500, 250), Image.ANTIALIAS)
            render2 = ImageTk.PhotoImage(load)
            img2 = tk.Label(self, image=render2)
            img2.image = render2
            img2.place(x=200, y=50)

            """
            def histo():
                load = Image.open('histo.png')
                width, height = load.size
                ratio = 2.5  # reduction ratio
                left = width / 15
                right = width
                top = height / 5
                bottom = 4 * height / 5
                load = load.crop((left, top, right, bottom))
                width, height = load.size
                load = load.resize((int(width / ratio), int(height / ratio)), Image.ANTIALIAS)
                render = ImageTk.PhotoImage(load)
                img.configure(image=render)
                img.image = render
                img.place(x=200, y=100)
            """

            eng = matlab.engine.start_matlab()
            # read original phantom
            if filetype == '.tif':
                if tifffile.imread(filepath).dtype == 'uint16':
                    y = eng.im2double(eng.loadtiff('test.tif'))
                else:
                    y = eng.im2double(eng.loadtiff(filepath))
            if filetype == '.txt':
                y = eng.open_reshape_txt(filepath)
            if n_crop_phantom.get() == 'True':
                y = eng.cropdata(y, 51, 125)

            sigma = int(sigma_ent.get()) / 100
            if n_variable_noise.get() == 'True':
                map = eng.helper.getNoiseMap(y, int(noise_factor_ent.get()))
            else:
                map = eng.ones(eng.size(y))

            eta = eng.times(sigma, map)
            if n_distribution.get() == 'Rice':
                z = eng.sqrt(eng.power(eng.plus(y, eng.times(eta, eng.randn(eng.size(y)))), 2) + eng.power(
                    eng.times(eta, eng.randn(eng.size(y))), 2))
            else:
                z = eng.plus(y, eng.times(eta, eng.randn(eng.size(y))))
            # print(z)
            print('Denoising Started')

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

            [y_est, sigma_est] = eng.bm4d(z, n_distribution.get(), eng.times(eng.minus(1, estimate_sigma), sigma),
                                          n_profile.get(), do_wiener, verbose, nargout=2)  # 1 - estimate_sigma to change boolean

            # plot histogram of the estimated standard deviation
            if n_estimated_sigma.get() == 'True':
                eng.helper.visualizeEstMap(y, sigma_est, eta, nargout=0)
                """
                global histo_btn
                histo_btn = tk.Button(self, text='Histogram', command=histo)
                histo_btn.place(x=370, y=300)
                
                
            elif n_estimated_sigma.get() == 'False':
                if 'histo_btn' in globals():
                    histo_btn.place_forget()
                else:
                    pass
                    """

            eng.helper.visualizeXsect(y, z, y_est, nargout=0)
            eng.y_est_to_tif(y_est, K, filetype, nargout=0)
            eng.quit()

            def sel(self):
                if filetype == '.tif':
                    img = tifffile.imread(filepath)
                    dst = np.zeros(shape=(img.shape[1], img.shape[2]))
                    data_raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
                elif filetype == '.txt':
                    data_raw = tifffile.imread('raw.tif')
                load_raw = Image.fromarray(data_raw[int(var.get())].astype('uint8'))
                width, height = load_raw.size
                ratio = M/200
                load_raw = load_raw.resize((int(width / ratio), int(height / ratio)), Image.ANTIALIAS)
                render_raw = ImageTk.PhotoImage(load_raw)
                img_raw.configure(image=render_raw)
                #img_raw = tk.Label(self, image=render_raw)
                img_raw.image = render_raw
                img_raw.place(x=220, y=60)

                data_denoise = tifffile.imread('denoise_bm4d.tif')
                load_denoise = Image.fromarray(data_denoise[int(var.get())].astype('uint8'))
                width, height = load_denoise.size
                ratio = M/200
                load_denoise = load_denoise.resize((int(width / ratio), int(height / ratio)), Image.ANTIALIAS)
                render_denoise = ImageTk.PhotoImage(load_denoise)
                img_denoise.configure(image=render_denoise)
                #img_denoise = tk.Label(self, image=render_denoise)
                img_denoise.image = render_denoise
                img_denoise.place(x=450, y=60)

            global var
            var = tk.DoubleVar(self)
            tk.Scale(self, variable=var, orient=tk.HORIZONTAL, from_=0, to=K-1, length=370,
                     command=sel).place(x=220, y=285)
            var.set(int(K/2))

            if filetype == '.tif':
                img = tifffile.imread(filepath)
                dst = np.zeros(shape=(img.shape[1], img.shape[2]))
                data_raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
            elif filetype == '.txt':
                data_raw = tifffile.imread('raw.tif')
            load_raw = Image.fromarray(data_raw[int(K/2)])
            width, height = load_raw.size
            ratio = M/200
            load_raw = load_raw.resize((int(width / ratio), int(height / ratio)), Image.ANTIALIAS)
            render_raw = ImageTk.PhotoImage(load_raw)
            #img_raw.configure(image=render_raw)
            global img_raw
            img_raw = tk.Label(self, image=render_raw)
            img_raw.image = render_raw
            img_raw.place(x=220, y=60)

            data_denoise = tifffile.imread('denoise_bm4d.tif')
            load_denoise = Image.fromarray(data_denoise[int(K/2)])
            width, height = load_denoise.size
            ratio = M/200
            load_denoise = load_denoise.resize((int(width / ratio), int(height / ratio)), Image.ANTIALIAS)
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
                root.geometry("300x120")
                root.title("Denoising Quality")
                x_position = 80
                eng = matlab.engine.start_matlab()
                if filetype == '.tif':
                    if tifffile.imread(filepath).dtype == 'uint16':
                        eng.sixteen_to_eight(filepath, nargout=0)
                        A = eng.imread('test.tif')
                        ref = eng.imread("denoise_bm4d.tif")
                        peaksnr = eng.peaksnr(A, ref)
                        ssimval = eng.ssim(A, ref)
                        rmse = eng.quality_check('test.tif', 'denoise_bm4d.tif')
                    else:
                        A = eng.imread(filepath)
                        ref = eng.imread("denoise_bm4d.tif")
                        peaksnr = eng.peaksnr(A, ref)
                        ssimval = eng.ssim(A, ref)
                        rmse = eng.quality_check(filepath, 'denoise_bm4d.tif')
                elif filetype == '.txt':
                    A = eng.imread('raw.tif')
                    ref = eng.imread("denoise_bm4d.tif")
                    peaksnr = eng.peaksnr(A, ref)
                    ssimval = eng.ssim(A, ref)
                    rmse = eng.quality_check('raw.tif', 'denoise_bm4d.tif')
                eng.quit()
                tk.Label(root, text='PSNR: '+str(peaksnr)).place(x=x_position, y=20)
                tk.Label(root, text='SSIM: '+str(ssimval)).place(x=x_position, y=50)
                tk.Label(root, text='RMSE: '+str(rmse)).place(x=x_position, y=80)

            tk.Button(self, text='Check Denoising Quality', command=check).place(x=475, y=365)

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
        n_estimated_sigma_chosen.set('False')
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

        """
        # Dropdown lists
        tk.Label(self, text='View Option:').place(x=208, y=12)

        n_view = tk.StringVar()
        n_view_chosen = ttk.Combobox(self, width=9, textvariable=n_view)
        n_view_chosen['values'] = ('Horizontal', 'Coronal', 'Sagittal')
        n_view_chosen.set('Horizontal')
        n_view_chosen.bind("<<ComboboxSelected>>", change_type)
        n_view_chosen.place(x=430, y=12)

        n_type = tk.StringVar()
        n_type_chosen = ttk.Combobox(self, width=8, textvariable=n_type)
        n_type_chosen['values'] = ('Raw', 'Noise', 'Denoised')
        n_type_chosen.set('Raw')
        n_type_chosen.bind("<<ComboboxSelected>>", change_type)
        n_type_chosen.place(x=350, y=12)

        n_dimension = tk.StringVar()
        n_dimension_chosen = ttk.Combobox(self, width=5, textvariable=n_dimension)
        n_dimension_chosen['values'] = ('2D', '3D')
        n_dimension_chosen.set('2D')
        n_dimension_chosen.bind("<<ComboboxSelected>>", change_type)
        n_dimension_chosen.place(x=287, y=12)
        """

        # Next and user manual
        tk.Button(self, text='Back', command=lambda: master.switch_frame(file_page)).place(x=10, y=365)
        tk.Button(self, text='User Manual', command=master.user_manual).place(x=60, y=365)
        tk.Button(self, text='Next', command=Next).place(x=630, y=365)



class denoising_stv(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master, height=800, width=800)  # need to define height and width to use .place()
        self.master.geometry('680x400')
        self.master.title('Chemical Image Analysis - Denoising - STV')
        img_x = 250
        img_y = 70
        global n
        n = 0

        def denoise_stv():
            global n
            n = 1

            eng = matlab.engine.start_matlab()
            eng.addpath(eng.genpath('./spectral_tv'))
            if filetype == '.tif':
                hyper_noisy = eng.read_hyperdata(filepath, M, N, K)
            elif filetype == '.txt':
                hyper_noisy = eng.open_reshape_txt(filepath)
            #[rows, cols, frames] = eng.size(hyper_noisy)

            beta = beta_ent.get().split(",")
            eng.make_beta_array(n_tv_method.get(), float(rho_r_ent.get()), float(rho_o_ent.get()), float(beta[0]), float(beta[1]), float(beta[2]), float(gamma_ent.get()), float(max_itr_ent.get()), float(alpha_ent.get()), float(tol_ent.get()), hyper_noisy, K, nargout=0)
            eng.quit()


            def sel(self):
                if filetype == '.tif':
                    img = tifffile.imread(filepath)
                    dst = np.zeros(shape=(img.shape[1], img.shape[2]))
                    data_raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
                elif filetype == '.txt':
                    data_raw = tifffile.imread('raw.tif')
                load_raw = Image.fromarray(data_raw[int(var.get())].astype('uint8'))
                width, height = load_raw.size
                ratio = M/200
                load_raw = load_raw.resize((int(width / ratio), int(height / ratio)), Image.ANTIALIAS)
                render_raw = ImageTk.PhotoImage(load_raw)
                img_raw.configure(image=render_raw)
                #img_raw = tk.Label(self, image=render_raw)
                img_raw.image = render_raw
                img_raw.place(x=220, y=60)

                data_denoise = tifffile.imread('denoise_stv.tif')
                load_denoise = Image.fromarray(data_denoise[int(var.get())].astype('uint8'))
                width, height = load_denoise.size
                ratio = M/200
                load_denoise = load_denoise.resize((int(width / ratio), int(height / ratio)), Image.ANTIALIAS)
                render_denoise = ImageTk.PhotoImage(load_denoise)
                img_denoise.configure(image=render_denoise)
                #img_denoise = tk.Label(self, image=render_denoise)
                img_denoise.image = render_denoise
                img_denoise.place(x=450, y=60)

            global var
            var = tk.DoubleVar(self)
            tk.Scale(self, variable=var, orient=tk.HORIZONTAL, from_=0, to=K-1, length=370,
                     command=sel).place(x=220, y=285)
            var.set(int(K/2))

            if filetype == '.tif':
                img = tifffile.imread(filepath)
                dst = np.zeros(shape=(img.shape[1], img.shape[2]))
                data_raw = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
            elif filetype == '.txt':
                data_raw = tifffile.imread('raw.tif')
            load_raw = Image.fromarray(data_raw[int(K/2)])
            width, height = load_raw.size
            ratio = M/200
            load_raw = load_raw.resize((int(width / ratio), int(height / ratio)), Image.ANTIALIAS)
            render_raw = ImageTk.PhotoImage(load_raw)
            #img_raw.configure(image=render_raw)
            global img_raw
            img_raw = tk.Label(self, image=render_raw)
            img_raw.image = render_raw
            img_raw.place(x=220, y=60)

            data_denoise = tifffile.imread('denoise_stv.tif')
            load_denoise = Image.fromarray(data_denoise[int(K/2)])
            width, height = load_denoise.size
            ratio = M/200
            load_denoise = load_denoise.resize((int(width / ratio), int(height / ratio)), Image.ANTIALIAS)
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
                root.geometry("300x120")
                root.title("Denoising Quality")
                x_position = 80
                eng = matlab.engine.start_matlab()
                if filetype == '.tif':
                    if tifffile.imread(filepath).dtype == 'uint16':
                        eng.sixteen_to_eight(filepath, nargout=0)
                        A = eng.imread('test.tif')
                        ref = eng.imread("denoise_stv.tif")
                        peaksnr = eng.peaksnr(A, ref)
                        ssimval = eng.ssim(A, ref)
                        rmse = eng.quality_check('test.tif', 'denoise_stv.tif')
                    else:
                        A = eng.imread(filepath)
                        ref = eng.imread("denoise_stv.tif")
                        peaksnr = eng.peaksnr(A, ref)
                        ssimval = eng.ssim(A, ref)
                        rmse = eng.quality_check(filepath, 'denoise_stv.tif')
                elif filetype == '.txt':
                    A = eng.imread('raw.tif')
                    ref = eng.imread("denoise_stv.tif")
                    peaksnr = eng.peaksnr(A, ref)
                    ssimval = eng.ssim(A, ref)
                    rmse = eng.quality_check('raw.tif', 'denoise_stv.tif')
                eng.quit()
                tk.Label(root, text='PSNR: '+str(peaksnr)).place(x=x_position, y=20)
                tk.Label(root, text='SSIM: '+str(ssimval)).place(x=x_position, y=50)
                tk.Label(root, text='RMSE: '+str(rmse)).place(x=x_position, y=80)

            tk.Button(self, text='Check Denoising Quality', command=check).place(x=475, y=365)

        def Next():
            if decomposition_method == 'MCR':
                master.switch_frame(decomposition_mcr)
            elif decomposition_method == 'Spectral Phasor':
                master.switch_frame(decomposition_phasor)
            elif decomposition_method == 'LS':
                master.switch_frame(decomposition_ls)


        """
        def change_type(self):
            if n == 0:
                pass
            else:
                if n_type.get() == 'Raw':
                    load = Image.open('stv_noise_sample.png')
                elif n_type.get() == 'Denoised':
                    load = Image.open('stv_denoise_sample.png')
                    # bottom = 2.6 * height / 4
                    # load = load.crop((left, top, right, bottom))
                    # width, height = load.size
                width, height = load.size
                ratio = 3.5  # reduction ratio
                load = load.resize((int(width / ratio), int(height / ratio)), Image.ANTIALIAS)
                render = ImageTk.PhotoImage(load)
                img.configure(image=render)
                # img = tk.Label(self, image=render)
                img.image = render
                img.place(x=img_x, y=img_y)
        """

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
        tk.Button(self, text='Next', command=Next).place(x=630, y=365)


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

        # Decomposition label
        tk.Label(self, text='Decomposition', font=(font, 14, 'bold')).place(x=10, y=10)

        # Separator
        separator1 = ttk.Separator(self, orient='horizontal')
        separator1.place(x=0, y=350, relwidth=1, relheight=1)

        tk.Button(self, text='Back', command=back).place(x=10, y=365)
        tk.Button(self, text='User Manual', command=master.user_manual).place(x=60, y=365)
        tk.Button(self, text='Next').place(x=630, y=365)


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
        tk.Button(self, text='Next').place(x=630, y=365)


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

        def decompose():
            eng = matlab.engine.start_matlab()
            #filepath = 'C:/Users/User/Dropbox/My PC (LAPTOP-7BC0EJ3C)/Downloads/LS_LASSO/Step size0.0040_Dwell time10 Celegans_Jian_1040-100_800-20_2.45MHz_R_OBJ5_P23460_F23070_2_up1um.txt'
            #filetype = '.txt'
            if denoising is False or n_use.get()==0:
                #filepath = 'C:/Users/User/Dropbox/My PC (LAPTOP-7BC0EJ3C)/Downloads/LS_LASSO/Step size0.0040_Dwell time10 Celegans_Jian_1040-100_800-20_2.45MHz_R_OBJ5_P23460_F23070_2_up1um.txt'
                input_file = filepath
            else:
                if denoising_method == 'Bm4d':
                    input_file = 'denoise_bm4d.tif'
                elif denoising_method == 'STV':
                    input_file = 'denoise_stv.tif'
            eng.least_square(input_file, spectralpath, float(peaks_ent.get().split(',')[0]), float(peaks_ent.get().split(',')[1]), float(shifts_ent.get().split(',')[0]), float(shifts_ent.get().split(',')[1]), float(level_ent.get()), nargout=0)
            eng.quit()

            global img_ls
            img = tifffile.imread('./ls_chemical_maps/'+n_component.get()+'.tif')
            load_raw = Image.fromarray(img)
            load_raw = load_raw.resize((int(400/1.1), int(300/1.1)), Image.ANTIALIAS)
            render_raw = ImageTk.PhotoImage(load_raw)
            img_ls = tk.Label(self, image=render_raw)
            img_ls.image = render_raw
            img_ls.place(x=60, y=70)

        def switch(self): # self is needed here
            img = tifffile.imread('./ls_chemical_maps/'+n_component.get()+'.tif')
            load_raw = Image.fromarray(img)
            load_raw = load_raw.resize((int(400/1.1), int(300/1.1)), Image.ANTIALIAS)
            render_raw = ImageTk.PhotoImage(load_raw)
            img_ls.configure(image=render_raw)
            img_ls.image = render_raw
            img_ls.place(x=60, y=70)

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
        tk.Label(self, text='Components:').place(x=490, y=180)
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
        tk.Button(self, text='Next').place(x=630, y=365)






if __name__ == "__main__":
    app = Main()
    #app.title("Denoising v0.1")
    app.mainloop()


