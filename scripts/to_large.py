from PIL import Image
import glob, os, sys

count = 1
max = len(glob.glob(sys.argv[1] + '*.png'))
for infile in glob.glob(sys.argv[1] + '*.png'):
    sys.stdout.write('\rRunning: ' + str(count) + '\\' + str(max))
    file, ext = os.path.splitext(infile)
    img = Image.open(infile)
    img = img.resize((1024, 1024))
    img.save(file + '-large' + ext)
    count = count + 1
print()

count = 1
max = len(glob.glob(sys.argv[1] + '*.jpg'))
for infile in glob.glob(sys.argv[1] + '*.jpg'):
    sys.stdout.write('\rRunning: ' + str(count) + '\\' + str(max))
    file, ext = os.path.splitext(infile)
    img = Image.open(infile)
    img = img.resize((1024, 1024))
    img.save(file + '-large' + ext)
    count = count + 1
print()