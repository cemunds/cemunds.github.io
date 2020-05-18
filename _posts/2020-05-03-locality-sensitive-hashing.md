---
title: "Locality-Sensitive Hashing for Image Deduplication"
categories: Posts
date: "11.05.2020"
---

A few months ago, I had a big image dataset of around 90.000 small images (less than 100x100 pixels) that I wanted to train an image classifier with. Unfortunately, there were a lot of duplicate and near-duplicate images in the set, which showed to have an effect on the trained classifier, making it overfit to certain images that appeared more often than others.

To counteract this effect, I researched on how to do image deduplication and soon came across *fingerprinting* and *locality-sensitive hashing*. The posts and articles I found were mainly concerned with the deduplication of text documents, which is why I want to summarize the insights I gained by applying these concepts to images in this post.

## Fingerprinting the images
Fingerprinting is concerned with calculating a hash value for each image. However, for the use case of finding duplicate images we need to use a perceptual hash function (in contrast to a cryptographic hash function). A function that is commonly used in this context is the difference hash (dHash). Given an image and the desired size of the hash, the dHash can easily be calculated in a few lines of Python:

```python
def dhash(image, k=8):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (k + 1, k))
    diff = resized[:, 1:] > resized[:, :-1]
    return np.array([1 if i else 0 for i in diff.flatten()])
```

The function converts the given image to grayscale, resizes it to <code>(k+1) x k</code>, and compares the pixel intensities of the resulting image to another version of the image, shifted one pixel to the right. This creates a 2D grid of bits, indicating at which positions the right pixel is brighter than the pixel to the left of it. For <code>k=8</code>, flattening the 2D grid to a 1D array results in a binary hash that consists of 64 (8²) zeroes and ones and that is invariant to overall image brightness and partly invariant to image scale.

{% include figure image_path="/assets/images/lsh-preprocessing.png" caption="[Figure 1] Preprocessing of an image from the [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html). In this example, I chose k=16, which would result in a 256-bit binary hash." %}

Now that we can calculate the fingerprint for all of our images, we need an efficient way of comparing them to each other to identify duplicates and images that are *almost* duplicates. In the case of a 64-bit hash, Dr. Neal Krawetz of [HackerFactor](http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html) suggests that hashes with a Hamming distance of more than ten bits likely indicate different images, while images with a distance smaller than ten are likely to be variations of the same image.

Of course we could just compare the fingerprint of each image to each other fingerprint. This brute force approach would certainly give you the exact result, identifying all the image pairs with a Hamming distance of less than the desired threshold. However, for a collection of *n* images we would need to perform <code>n*(n-1)/2</code> comparisons, which indicates a runtime of O(n²). Just out of curiosity I ran the brute force algorithm on my image set of 90.000 images over night. The program ran for roughly 15 hours, so this is definitely not the way to do it (especially since images were going to be added to my dataset over time and I would have to run the deduplication process every so often as a kind of "housekeeping" process).

## Locality-sensitive hashing to the rescue
Locality-sensitive hashing (LSH) is an approximate algorithm to find nearest neighbours. An approximate algorithm won't find **all** the duplicate images in the set, but it is tremendously faster than the brute force approach. Additionally, it can be shown mathematically that the rate of false positives and false negatives lies within a certain range.

In LSH, the image fingerprints we calculated earlier will be sorted into buckets in a way that images that are similar are likely to end up in the same bucket, while images that are different are likely to be put into different buckets.

The method we use to sort the fingerprints into buckets depends on the distance measure we want to approximate:

* Use Bit sampling to approximate the Hamming distance.
* Use MinHash to approximate the Jaccard similarity (useful when working with text documents).
* Use the random projection method when approximating the cosine distance between vectors.

Since we want to use the Hamming distance to compare the fingerprints, we will have to use the [Bit sampling](http://mlwiki.org/index.php/Bit_Sampling_LSH) method.

For the Bit sampling method, we sample *L* subsets (I<sub>1</sub>, ..., I<sub>L</sub>) of size *k* uniformly with replacement from the range [0, d), where d is the dimensionality of our fingerprint. Let *h* be our fingerprint, then the function g<sub>j</sub>(h) projects h on I<sub>j</sub> by selecting bits out of *h* according to I<sub>j</sub> and concatenating them to a new hash of size *k*.

In the next step, the projected hashes themselves are stored in another hash table mapping a projected hash to a list of image fingerprints that were projected onto this hash. As we repeat this procedure, it will eventually happen that two image fingerprints are mapped to the same field in our hash table of projected hashes. These images then form a candidate pair and we can take their corresponding fingerprints to test if they really have a Hamming distance below the desired threshold, or if they were just mapped onto the same field by coincidence.

Let's look at an example. Suppose we have three 16-bit image fingerprints h<sub>1</sub>, h<sub>2</sub> and h<sub>3</sub>:
<pre>
h<sub>1</sub> = 1010101101010001
h<sub>2</sub> = 1010101001001101
h<sub>3</sub> = 0000010010100010
</pre>

We then sample <code>L = 3</code> subsets of size <code>k = 4</code> from the range [0, 16):
<pre>
I<sub>1</sub> = [4, 2, 7, 5]
I<sub>2</sub> = [3, 11, 6, 4]
I<sub>3</sub> = [3, 13, 4, 6]
</pre>

Now we project each image fingerprint onto I<sub>1</sub>, ..., I<sub>3</sub> by selecting the corresponding bits and concatenating them together.
<pre>
g<sub>1</sub>(h<sub>1</sub>) = 1110
g<sub>1</sub>(h<sub>2</sub>) = 1100
g<sub>1</sub>(h<sub>3</sub>) = 0001

g<sub>2</sub>(h<sub>1</sub>) = 0111
g<sub>2</sub>(h<sub>2</sub>) = 0011
g<sub>2</sub>(h<sub>3</sub>) = 0000

g<sub>3</sub>(h<sub>1</sub>) = 0011
g<sub>3</sub>(h<sub>2</sub>) = 0111
g<sub>3</sub>(h<sub>3</sub>) = 0000
</pre>

You see that g<sub>2</sub>(h<sub>1</sub>) and g<sub>3</sub>(h<sub>2</sub>) result in the same hash. Another pair would be g<sub>2</sub>(h<sub>2</sub>) and g<sub>3</sub>(h<sub>1</sub>), but they constitute the same image candidate pair, namely image 1 and image 2. To make sure that the images 1 and 2 really are close enough in Hamming space, we calculate the distance between their full-length fingerprints.

<pre>
1010101101010001
1010101001001101
----------------
0000000100011100
</pre>

We see that four out of 16 bits are different. To conclude whether a distance of four bits given 16-bit hashes is close enough to indicate similar images takes some experimentation. Just as a reminder, in the example above it was stated that, in the case of 64-bit hashes, a difference of less than ten bits is likely to indicate a variation of the same image.

Finally, here is the code for the class that is responsible for the deduplication. The full code can be found on my [GitHub](https://github.com/cemunds/near-image-deduplication):

```python
class LSHDeduplicator(Deduplicator):
    """
    A Deduplicator that uses Locality-Sensitive Hashing to identify potential duplicate images. The LSH algorithm
    is orders of magnitudes faster than the brute force approach. However, it only returns an approximate
    result and might not find certain duplicates.
    """

    def __init__(self, hash_func=dhash, k=32, l=50, d=64):
        """
        Constructs an LSHDeduplicator.
        :param hash_func: The hash function to be used for fingerprinting the images.
        :param k: Length of the projected hash.
        :param l: The number of hash functions to use in the LSH algorithm.
        :param d: The dimensionality of the image hash.
        """
        Deduplicator.__init__(self, hash_func)
        self._k = k
        self._l = l
        self._d = d
        self._projections = np.array([np.random.choice(list(range(d)), k) for _ in range(l)])

    def deduplicate(self, imgs):
        """
        Deduplicates the given images.
        :param imgs: Images to deduplicate.
        :return: A list of lists. Detected duplicate images are grouped together in one list and there
        is one list per detected group.
        """
        hash_table = {}
        result = []
        duplicate_count = 0

        for img in imgs:
            hashed_img = HashedImage(img, self._hash_func(img), [])
            unique = True
            for g in self._projections:
                g_x = "".join(hashed_img.hash[g].astype("str"))
                l = hash_table.get(g_x, [])

                for potential_duplicate in l:
                    if not unique:
                        break

                    if hashed_img is potential_duplicate:
                        continue

                    distance = hamming(hashed_img.hash, potential_duplicate.hash) * self._d
                    if distance < 10:
                        unique = False
                        potential_duplicate.duplicates.append(hashed_img)
                        duplicate_count += 1

                l.append(hashed_img)
                hash_table[g_x] = l

            if unique:
                result.append(hashed_img)

        result = [self._flatten(img) for img in result]
        return result
```

## Conclusion
The LSH alrogithm is an efficient approximate algorithm to identify nearest neighbors. In my use case, it helped me clean up my dataset (which unfortunately I am not allowed to share) significantly, thus also improving the performance of the image classifier that was trained on this data.

## References
1. <https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134>
2. <https://towardsdatascience.com/fast-near-duplicate-image-search-using-locality-sensitive-hashing-d4c16058efcb>
3. <https://en.wikipedia.org/wiki/Locality-sensitive_hashing>
4. <https://www.pyimagesearch.com/2017/11/27/image-hashing-opencv-python/>
5. <https://realpython.com/fingerprinting-images-for-near-duplicate-detection/>