import matplotlib 
matplotlib.use('Qt4Agg')
import pyfits
image = pyfits.open('/project/ch2/RADIO_MAPS/out/21247.fits')
data = image[0].data
print(data)
import matplotlib.pyplot as plt
plt.imshow(data)
plt.show()

