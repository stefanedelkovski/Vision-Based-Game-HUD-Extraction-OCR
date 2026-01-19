## What's new?

-   NodeJS scraper will output images only of the video stream area and not the whole browser
-   New parameter value is introduced **--output=stdout**. Using this value the script will write the image data to process.stdout instead saving the images to the disk
-   New parameter introduced **--date-subdir(default: yes)**. This parameter works with --output=/abs/path/to/images-directory and if set to "no" will not generate child subdirectory where the images will be stored as the old behavior

## NodeJS script options

Run the following command and script help will be printed

```
node index.js
```

## How to convert image_data to cv2.imread()

```
import numpy as np
import cv2
import base64

# Assuming `encoded_data` is your base64 string
image_data = base64.b64decode(encoded_data)

# Convert the binary data to a NumPy array
nparr = np.frombuffer(image_data, np.uint8)

# Decode the image array into an actual image
image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # or cv2.IMREAD_GRAYSCALE depending on your needs

# Now `image` is a NumPy array similar to what you'd get from cv2.imread()
print(type(image))  # <class 'numpy.ndarray'>
print(image.shape)  # e.g., (480, 640, 3)
print(image.dtype)  # uint8

```

## Notes

There is part of the presentation where it was mentioned that `processed_batches` are almost equal to `total_batches` and this was said to be a minus of the current approach.

_Answer:_ this was happening, because the previous logic was producing frames way more faster than the current approach. If you want to replicate the previous behavior you can set **--interval-ms** to a small value like **--interval-ms=10**. Then `total_batches` will become much higher that `processed_batches` and the processing model will not be able to catchup with the `capture_images` logic. But honestly, I think this is not okay, because at some point the program will run into **out of memory exception** and will crash. I think you would like to try keep the `processed_batches` and `total_batches` with similar values, if you want this program to run for a long period of time.

There is another part of the presentation where high CPU of the chrome headless browser is a minus.

_Answer:_ If you see closely there are also other processes that consume a lot of resources like `apache2` and `tesseract`. This is just the normal linux behavior and I don't see it as a minus. If there are multiple processes which spawn headless chrome browsers, the linux system will distribute it's resources between them and everything will work just fine. Linux is smart enough to do that. And about how many streams could be processed at the same time, this is a matter of testing, the machine needs to be pushed to its limits. The good thing is that this is the right time to do it :)

There is another statement in the presentation which says that if the models are being re-trained, the program will not be able to work and vice versa.

_Answer:_ Again, I think that linux is smart enough to distribute resources between the processes. Is this ever tested or this is only an assumption? I think there should be a logic to re-train the models on the server and I think this will not cause the other program to stop working while the re-training is going on. But this is my assumption, based on my experience of using linux servers, this needs to be tested and then any conclusions/statements could be done.
