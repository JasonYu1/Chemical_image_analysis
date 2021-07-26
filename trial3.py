import matlab.engine

filepath = 'beads_noisy.tif'
eng = matlab.engine.start_matlab()
A = eng.size(eng.imread(filepath))
#K = eng.length(eng.imfinfo(filepath))
print(matlab.double([0.1,0.2]))
eng.quit()