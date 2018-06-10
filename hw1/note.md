# 读书笔记

2015011313 徐鉴劲 计54

## A Survey of Shape Feature Extraction Techniques

### Shape Descriptors' Requirements

This paper proposes that shape descriptors must have some essential characteristics:

```
1. Identifiability.

2. Invariance in affine transformation.

3. Noise invariance.

4. Occulusion invariance.

5. Statistically independent.

6. Reliable.
```

But I don't think forcing a descriptor to match all the requirement is a good idea. E.g., statictiscal independent is not a good idea in modern deep learning feature extraction pipeline.

### Shape Parameters

Some simple geometric characteristic.

```
1. Centric of gravity. E.g., the centroid of contour can be computed.

2. Axis of least inertia.

3. Average bending energy.

...
```

### One-dimensional function for shape representation

In other word, there are some methods that try to describe the shape using a scalar function. The function usually takes the sequence of boundary points as input. It is also called shape signature.

```
1. Complex coordinates.

2. Centroid distance function.

3. Tangent angle (tangential direction of a contour).

...
```

### Polygonal approximation

Polygonal approximation aimed at ignoring minor variations along the edge, and capture the overall shape. In general, there are two categories, merging or splitting. This method can be used as preprocessing.

#### Merging methods

Merging methods add successive pixels to a line segment if each new pixel that is added doesn't cause the segment to deviate too much from a straight line.

1. Distance threshold. Connecting two relatively far points on the contour, once a distance error exceed threshold, connect this line and start a new one.

2. Tunneling method. Suitable for thick contours.

3. Polygon evolution. Substitude order is important.

#### Splitting method

Repeatively splitting a line into more segments.

### Spaticial Interrelation Feature

1. Adaptive Grid Resolution. Normalize orientation and then do a quad-tree decomposition.

2. Bounding box. First normalize oritation, and set up an initial bounding box. Then recurrently divide evenly the bounding box vertically, compress the bounding box. And then divide each bounding box horizontally and compress the bounding box.

3. Convex hull. Divide a shape into a concavity tree.

4. Chain code. Represent the boundary using a unity length and direction vector.

5. Basic chain code. Four-connectivity or 8-connectivity.

6. Differential chain code. Encode the difference.

7. Smooth curve decomposition.

8. Beam angle statistics.

9. Shape matrix. Square shape matrix or polar shape matrix.

10. Shape context. 

...

### Moments

All kinds of moments definition originates from physics.

### Scale space approaches

1. Curvature scale shape. Convolve using gaussian kernel. Smooth the curve iteratively.

2. Intersection point map.

### Shape transformation domain

## Image Segmentation Using Deformable Models

Deformable models consists of external force and internel force. The former serves as the pusher constraining the model to object boundaries and the latter serves as the maintainer for preserving smoothness etc. The deformable models can be divided into parametric ones and geometric ones. The former allows direct interaction with model and can lead to a compact representation. The latter is better at handling topological change such as splitting and merging.

### Parametric deformable models

The whole system of a deformable model is a curve under some forces. In all of the forces, potential force is from potential energy field from image. The energy field is defined simply as the gradient (edge map) of image after gaussian smoothing. Tension force is internal smooth conditions, formed by regulation of one-order and two-order derivatives. 

Generalizing to other forcing like dampening force, pressure force, the deformable model morphs into dynamic system. A lot of other forces are added, in which interactive force is the most interesting. This force is formed by and artificially placed landmarks or attracting/expelling point. Simply add another force to the equation works.

In practice, differential equations are transformed into difference equation. There are 2 disadvantages: 1. initial position matters. The solution is reparametrization, but it is computationally heavy. 2. Hard to deal with splitting and merging.

### Geometric deformable models

Geometric deformable model originate from curvature evolution theory. This theory focus on a evolution equation: Curvature of a curve push the curve to evolve, and the evolution speed is controled by another function. By using level set, image data in incorporated into original theory.

This method formalize curvature as level set. A level set is defined based on a level function. The set of points having the same value is level set. In geometric deformable model, zero level set is used. Level function is devised so that initial contour is exactly zero, a common choice is a specially devised distance function.

The curve evolution function is rewrited using level function, and image data is modeled as energy function to prevent the curvature from being expanded too much.

The main disadvantage of this method will be the inability to cope with noise.

### Extensions

1. Deformable Fourier model. In original model, curves are directly encoded using parameter function. In this method, the parameter function is decomposed and truncated into Fourior series. The representation is more concise.

Interestingly, human sketch can be applied using Bayesian prior.

## Conclusion

This two papers give concrete examples of handcrafted features. Computer scientists think mainly in physics or math aspects to mine the characteristics of objects. However, in practice this feature seems to be useless because of noise, occulusion, scale and various appearance and textures. I think this is because current human knowledge has not understood the mechanism of human vision and current model are far from the real mechanism.

On the contrary, recent research refer to representation learning instead of handcrafted features. Though the learned feature are also not understood, but the overall system proceeds well.