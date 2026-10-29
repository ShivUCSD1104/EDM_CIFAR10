---
noteId: "44827e90d33b11f1b1ae59132f567d77"
tags: []

---

# Notes

Answers to the questions in `dsc180-d32-cifar10-edm-lab/cifar10_edm_lab.ipynb`. The notebook marks several of these "Before running": write that prediction before looking at the figure.

## 1. Unconditional samples

Trajectory figure: the noisy image and `D(x; σ)` from `σ = 80` down to `σ = 0.002`.

### What will `D(x; σ)` look like at `σ = 80`?



### At which noise level do you expect the content of the final image to be settled?



## 2. Conditional samples

One class per row. Each column starts from the same noise.

### What do you expect to stay the same down a column?



## 3. CFG with two networks

`D_w(x; σ, c) = D(x; σ) + w (D(x; σ, c) − D(x; σ))`. The grid is CFG toward cat, `w` from 1 to 4, same starting noise in each column.

### What should change as `w` goes from 1 to 4?



### Trajectories with a condition

Same class and same seed for the conditional model (`w = 1`) and for CFG at `w = 4`.

#### At which noise level does the class first show in `D(x; σ)`? Is it earlier with `w = 4`?



#### From which noise level on does the image barely change?



#### How do the `w = 4` estimates differ from the `w = 1` ones at low noise?



## 4. FID in two dimensions

Three panels: same distribution with 30 points, a shifted mean, and too little variety.

### Which panel has the largest FID, and which term causes it?



## 5. The CFG sweep

Balanced class requests, same noise for every `w`. At `w = 0` the request is ignored.

### How do you expect accuracy and FID to move as `w` grows?



### Which `w` would you choose, and what does each measure miss?

Look at the images, not only the numbers.



## 6. Training on 100 and 1,000 images

### What does FID do across the snapshots, and why?



### Does `w > 1` raise label accuracy here?

Compare with part 5.



### Optional: how early does `D(x; σ)` lock onto one training image?

From a trajectory of this network.
