# 2-points method (Planar Motion)

Implmentation of essential matrix estimation under planar motion constraint. The implmentation based on 2-points method (ellipse-cirlce intersection) which is demonstrated in [Choi's paper: Fast and reliable minimal relative pose estimation under planar motion](https://www.sciencedirect.com/science/article/abs/pii/S0262885617301233).

## Evaluation
The implementation is evaluated using synthetic data which contains 20% outliers.

## Run command
3 parameters are needed:
- First image feature points file.
- The correspondence feature points in the second image file.
- The camera matrix file.

```sh
./build.sh

./build/my_app data/src_points_1920_1200_0.200000_0.200000_0.000000_2.000000_1.000000_0.000000_-2.000000_0.000000_-0.700000_0.000000.txt data/dest_points_1920_1200_0.200000_0.200000_0.000000_2.000000_1.000000_0.000000_-2.000000_0.000000_-0.700000_0.000000.txt data/K.txt
```