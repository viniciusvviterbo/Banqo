![Banqo](https://user-images.githubusercontent.com/24854541/105562486-142c6d00-5cf9-11eb-93ec-0ee7bb202d1d.png)

Recently I came across a quite curious question:

> How many places to sit are there in London?

Even if the answer shows no immediate use in a real-world scenario, it got me thinking. I was intrigued and now *had to know* how I could answer it.

The only problem is that this question does not specify *which* London.

It could be the [biplane flying boat](https://en.wikipedia.org/wiki/Saro_London), the [ship](https://en.wikipedia.org/wiki/SS_London_(1864)), the [company](https://en.wikipedia.org/wiki/London_Drugs), the [music label](https://en.wikipedia.org/wiki/London_Records), or the [asteroid](https://en.wikipedia.org/wiki/8837_London). But even if you stick to the city of London, you still have a [list of cities with the same name](https://londonist.com/london/features/places-named-london-that-aren-t-the-london).

That's why I decided to pick a more interesting definition: the [2005 american movie](https://www.imdb.com/title/tt0449061/), starring ~Captain America~ Chris Evans, Jessica Biel and Jason Statham.

Had I ever heard of this movie? No. Does it look like I'd watch it? Also no.

But that's the reason we automate things, right? Do stuff we don't want to.

So, as any other person would do, I implemented a program that uses deep learning-based object detectors (in this case, Single Shot Detectors) and a pre-trained convolutional neural network model (Caffe) to **detect chairs and sofas, track those objects, and count every time one of those items appeared on the screen** playing whichever video you chose.

## Setting the Environment

1. Clone this repository
```shell
git clone github.com/viniciusvviterbo/banqo
cd banqo
```
2. [Install dlib in your computer](https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/)
```shell
yay -S python-dlib
```
3. (Optional) Create and activate a virtual environment 
```shell
virtualenv .venv
source .venv/bin/activate
```
4. Install the project dependencies
```shell
pip3 install -r requirements.txt
```

## Usage
```shell
python banqo.py [-h] [-p PROTOTXT] [-m MODEL] [-i INPUT] [-o OUTPUT] [-c CONFIDENCE] [-s SKIP_FRAMES]
```

- PROTOTXT: Path to the Coffe prototxt file;
- MODEL: Path to the pre-trained CNN model;
- INPUT: Path to the input video. If not informed, the program will use the webcam as a data source;
- OUTPUT: Path to the output video. If not informed, the program will not register any of the video generated;
- CONFIDENCE: Minimum probability threshold which helps to filter out weak detections.;
- SKIP_FRAMES: Number of frames to skip between detection phases;

Example:
```shell
python3 banqo.py --input videos/example_video.mp4
```
```shell
python3 banqo.py \
--prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
--input input/example_video_in.mp4 \
```

## References

[pyimagesearch/ module](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/)

[mobilenet_ssd/ Caffe deep learning model files](https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/)


![Divisor - Banqo](https://user-images.githubusercontent.com/24854541/105561762-8a7ba000-5cf6-11eb-9c1e-d19a6b190222.png)

**[GNU AGPL v3.0](https://www.gnu.org/licenses/agpl-3.0.html)**
