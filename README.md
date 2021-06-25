# Pixelate
A Project-Based On Image processing And Dijkstra Path Finding Algorithm

This project is based on the instructions given in the following [Problem Statement](https://drive.google.com/file/d/1XZivQZIc6szvCmp2vMksxlliCC4whAkB/view?usp=sharing).

## Installation Guidelines
Along with this repository, it is needed to have the following repositories:
1. [Pixelate_Sample_Arena](https://github.com/Robotics-Club-IIT-BHU/Pixelate_Sample_Arena)
2. [Pixelate_Main_Arena](https://github.com/Robotics-Club-IIT-BHU/Pixelate_Main_Arena)

Follow the steps given in these repositories and install the packages required.


## Approach
1. The arena was converted into a 2D matrix using image processing techniques where a particular node number denoted each square of the arena
2. Dijkstra Path Finding Algorithm was used to determine the shortest path to the destination node; here, we used manhattan distance as the heuristic measure
3. Movement through one-ways was considered by disconnecting it from the graph whenever required
4. We used the differential drive to run the bot more efficiently

## Features
1. Visual representation of the arena and the bot movements were done using **PyBullet**
2. **Image processing** techniques were used to manipulate the data, i.e., shape, colour and aruco marker detection in the programmable form
3. **Dijkstra Path Finding Algorithm** determined the shortest path to reach the destination node

## References
1. [Run-on Pixelate_Sample_Arena](https://drive.google.com/file/d/1UvjsFBCqCi7iOGJ31PaYBjCLJ4yCy9GN/view?usp=sharing)
2. [Run-on Pixelate_Main_Arena](https://drive.google.com/file/d/1swEE_Imy73od_K_-a2RPHQ5FvANhRL1x/view?usp=sharing)
