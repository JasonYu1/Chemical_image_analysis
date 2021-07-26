from tkinter import *
from tkinter import ttk
from queue import Queue
import threading
import time

queue = Queue()
root = Tk()

# Function to do 'stuff' and place object in queue for later #
def foo():
    # sleep to demonstrate thread doing work #
    time.sleep(5)
    obj = [x for x in range(0,10)]
    queue.put(obj)

# Create thread object, targeting function to do 'stuff' #
thread1 = threading.Thread(target=foo, args=())

# Function to check state of thread1 and to update progressbar #
def progress(thread, queue):
    # starts thread #
    thread.start()

    # defines indeterminate progress bar (used while thread is alive) #
    pb1 = ttk.Progressbar(root, orient='horizontal', mode='indeterminate')

    # defines determinate progress bar (used when thread is dead) #
    pb2 = ttk.Progressbar(root, orient='horizontal', mode='determinate')
    pb2['value'] = 100

    # places and starts progress bar #
    pb1.pack()
    pb1.start()

    # checks whether thread is alive #
    while thread.is_alive():
        root.update()
        pass

    # once thread is no longer active, remove pb1 and place the '100%' progress bar #
    pb1.destroy()
    pb2.pack()

    # retrieves object from queue #
    work = queue.get()
    return work

work = progress(thread1, queue)
root.mainloop()