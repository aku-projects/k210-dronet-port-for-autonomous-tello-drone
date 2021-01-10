# K210-dronet port for autonomous tello drone

## **Work in Progress**
The port was halted due to an impediment. Please check https://github.com/sipeed/MaixPy/issues/360 for details
## Intro

Please follow the link to the original work in the acknowledgement below.

The idea was to run the Dronet on K210, generating RC commands that is sent to tello via ESP32.

## Work till date

The network architecure was slightly modified to take 200x200 grayscale images directly. Best weights were used that was provided in the original work.

A kmodel was generated that works as expected during inference on PC but the same doesn't work on the K210 chip on maix bit board

## Repo Contents
1) dronet - contains best weights from original work
2) images - 200x200 grayscale images used for testing
3) maixpyscript-output - current outputs on k210
4) output - binary output from ncc infer on PC
5) commands.txt - ncc commands used to a) convert tflite model to kmodel b) Infer using kmodel on PC for testing
6) dronet_modifier.py - Makes minor changes to existing dronet model to pass 200x200 grayscale images directly
7) inferencevalues.txt - contains values from original dronet model and kmodel inferences for comparison
8) maixpyscript.[]()py - example maixpy script
9)model.kmodel - kmodel generated using ncc
10)model.tflite - tflite generated using modifier python script
11) readbin.[]()py - read ncc infer binary outputs in output folder

## Acknowledgment
Based on the work - DroNet: Learning to fly by driving

https://github.com/uzh-rpg/rpg_public_dronet


    @article{Loquercio_2018,
        doi = {10.1109/lra.2018.2795643},
        year = 2018,
        author = {Antonio Loquercio and Ana Isabel Maqueda and Carlos R. Del Blanco and Davide Scaramuzza},
        title = {Dronet: Learning to Fly by Driving},
        journal = {{IEEE} Robotics and Automation Letters}
    }



