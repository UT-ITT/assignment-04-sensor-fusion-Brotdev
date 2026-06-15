[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AktWbCri)
# assignment-04-CV-Sensor-Fusion

## Task 1 - Perspective Transformation

```
Usage: python image_extractor <input> <output> <width>x<height>
```

The points can be clicked in clockwise order starting from the top left.
After all 4 points have been selected the image will be transformed.
Using `s` the image can then be saved or reset using `<ESC>`.

By pressing `q` the program can be exited

## Task 2 - AR_Game

Simply point the Aruco-Board at the camera and position your finger on the canvas.

The game is a simple flock simulation with particles, where each particle can
be destroyed with your finger.<br>
The game is over once all particles have been destroyed. 

By pressing `q` the program can be exited

## Task 3 - Sensor Fusion

Using the transformation from Task 2, the board is mapped to the camera dimensions.
The program tries to predict the position of Marker 5 using the accelerometer data
from a DIPPID Sensor based on a runtime changable value (alpha).

Simply point the camera at a Aruco-Board and move a Marker with id 5 around.

Controls:<br>
`UP`:    Increase alpha by 0.01<br>
`RIGHT`: Increase alpha by 0.1<br>
`DOWN`:  Decrease alpha by 0.01<br>
`LEFT`: Decrease alpha by 0.1<br>
`Q`: Exit application
