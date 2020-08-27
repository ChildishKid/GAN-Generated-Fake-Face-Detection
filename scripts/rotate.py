from PIL import Image
import glob, os, sys, random

count = 1
max = len(glob.glob(sys.argv[1] + '*.png'))
for infile in glob.glob(sys.argv[1] + '*.png'):
    sys.stdout.write('\rRunning: ' + str(count) + '\\' + str(max))
    file, ext = os.path.splitext(infile)
    img = Image.open(infile)
    if random.randrange(0, 10) > 4:
        img = img.rotate(15)
    else:
        img = img.rotate(-15)
    img.save(file + '-15' + ext)
    count = count + 1
print()

count = 1
max = len(glob.glob(sys.argv[1] + '*.jpg'))
for infile in glob.glob(sys.argv[1] + '*.jpg'):
    sys.stdout.write('\rRunning: ' + str(count) + '\\' + str(max))
    file, ext = os.path.splitext(infile)
    img = Image.open(infile)
    if random.randrange(0, 10) > 4:
        img = img.rotate(15)
    else:
        img = img.rotate(-15)
    img.save(file + '-15' + ext)
    count = count + 1
print()