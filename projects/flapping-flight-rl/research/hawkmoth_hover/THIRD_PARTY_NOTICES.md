# Third-party notices

## IBAMR example-derived application structure

`src/main.cpp` is adapted from the IBAMR 0.19.0 `examples/IB/explicit/ex7/example.cpp`, including its IBAMR initialization and hierarchy setup patterns. IBAMR is distributed under the BSD 3-Clause license. The upstream copyright notice is retained at the top of the source file. Upstream project: [IBAMR](https://github.com/IBAMR/IBAMR); source tag `v0.19.0`, commit `a0a8d0a4dc5deb6f576a5837b211baf0d99f5df6`.

The code in this repository adds the hawkmoth kinematics, paired target-point motion, control-volume force sampling, and run outputs; it should not be represented as upstream IBAMR code.

## IBAMR 0.19.0 staggered Navier–Stokes regression fixture

`reference_data/ibamr_0.19.0_navier_stokes/` includes the upstream `navier_stokes_01.cpp` test source, its 3-D input deck, and its expected output from IBAMR 0.19.0 (commit `a0a8d0a4dc5deb6f576a5837b211baf0d99f5df6`). These files are distributed under the same BSD 3-Clause license. The original copyright/license header remains in the C++ source; see the IBAMR project and license text linked above.
