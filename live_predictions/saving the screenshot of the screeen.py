# import pyscreenshot as ImageGrab

# grab fullscreen
im = ImageGrab.grab()

# save image file
im.save('fullscreen.png')



import pyscreenshot as ImageGrab

# part of the screen
im = ImageGrab.grab(bbox=(10, 10, 510, 510))  # X1,Y1,X2,Y2

# save image file
im.save('box.png')

# You can force a backend
import pyscreenshot as ImageGrab
im = ImageGrab.grab(backend="scrot")
im.save('backend.png')
# You can force if subprocess is applied, setting it to False together with mss gives the best performance:

from time import time
# best performance
import pyscreenshot as ImageGrab
t0 = time()
for i in range(100):
    im = ImageGrab.grab(backend="mss", childprocess=False)
    # im.save('best_performance'+str(i)+'.jpg')
print(time()-t0)